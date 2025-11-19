from enum import Enum
import numpy as np
from numpy import cosh, sinh, exp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation


#  THE VISION:
#
#  ----------------------------------------------------------------
#  |                                                              |
#  |  --------------------------------------------------  ------  |
#  |  |                                                |  |    |  |
#  |  |                                                |  |  D |  |
#  |  |                                                |  |<-->|  |
#  |  |                                                |  ------  |
#  |  |                                                |          |
#  |  |                                                |  ------  |
#  |  |                                                |  |  T |  |
#  |  |                                                |  |  E |  |
#  |  |                   SIMULATION                   |  |  M |  |
#  |  |                                                |  |  P |  |
#  |  --------------------------------------------------  ------  |
#  |                                                              |
#  ----------------------------------------------------------------

###################
#    CONSTANTS    #
###################
GRID = 32
TOTAL_SPINS = GRID**2

CURRENT_D = 2  # this will be a single source of truth for everything
BETA = 1
MAX_D = 5  # I'm not sure how insightful going much higher than 4 is, and I'm not sure I could display the spins in a visually sensible way
MAX_BETA = 4*CURRENT_D  # how far the beta slider should go

# What's the intuition behind this choice? Maybe I want every vertex updated every second on average or something?
INTERVAL = 1 # 1000 / GRID**2

class GraphGeometry(Enum):
    COMPLETE = 1
    LATTICE = 2

CURRENT_GRAPH = GraphGeometry.COMPLETE

####################
# HELPER FUNCTIONS #
####################
colors_simple = [
    '#FF00FF',  # -5: Magenta
    '#00FFFF',  # -4: Cyan 
    '#FFA500',  # -3: Orange
    '#0080FF',  # -2: Blue
    '#FFFF00',  # -1: Yellow
    '#FFFFFF',  #  0: Black (unused)
    '#00FF00',  #  1: Green
    '#FF0000',  #  2: Red
    '#0000FF',  #  3: Dark Blue
    '#FF8C00',  #  4: Dark Orange
    '#800080',  #  5: Purple
]

# Create colormap
cmap_simple = ListedColormap(colors_simple)

# boundary norm so that my colours go exactly where I want
boundaries = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
boundary_norm = BoundaryNorm(boundaries, cmap_simple.N)

def unif_ising():
    return np.random.choice([-1,1])

def proportions_from_state_unormalised(state):
    """Given a state, counts the number of spins of each kind. Returns a proportions vector of the form (#e1, #e2, ..., #-e1, ... #-eMAXD)."""
    values, counts = np.unique(state, return_counts=True)

    prop = [0 for _ in range(2*MAX_D)]
    for count_index, spin in enumerate(values):
        prop_index = get_prop_index(spin)
        prop[prop_index] = counts[count_index]
    
    return np.array(prop) 
    # Note that the simulation could be made more efficient by keeping track of the proportions vector with each update, instead of counting everything each step.

def magnetisation_from_propotion(proportion):
    return np.array([proportion[i] - proportion[i+MAX_D] for i in range(MAX_D)])

def get_prop_index(spin):
    """
    Given a spin -MAX_D, ..., -1, 1, ..., MAX_D, returns the index of a proportions vector that corresponds.
    Recall proportions vectors are of the form (#e1, #e2, ..., #-e1, ... #-eMAXD).
    """
    if spin > 0 :  # just need to index from zero
        return spin - 1
    else:  # need to start from MAX_D
        return MAX_D + (- spin) - 1

def get_spin(prop_index):
    """
    Given a proportions index, return the corresponding spin
    """
    if prop_index < MAX_D:  # spin is positive
        return prop_index + 1
    else:  # spin is negative
        return MAX_D - prop_index - 1


def get_magnetisation(mag, spin):
    """
    Given a magnetisation vector and a spin, returns the magnetisation in the corresponding direction
    """
    if spin > 0:  # just need to index from zero
        return mag[spin-1]
    else:  # if the spin is negative, return whatever is in the vector *-1
        return - mag[abs(spin)-1]

def get_prop_update(spin):
    """
    Given a spin, generates a vector with a 1 in the corresponding position. Intended use is to update proportions
    """
    res = np.zeros(MAX_D*2)
    if spin > 0:  # just need to count from zero
        res[spin-1] = 1
    else:  # start from MAX_D
        res[MAX_D + (spin*-1)] = 1

    return res

def get_spin_vector(spin):
    """
    Convertes a spin (represented by an integer) to the corresponding np vector
    """
    index = abs(spin)-1
    return np.eye(1, MAX_D, index) * np.sign(spin)


##############
# SIMULATION #
##############
def select_vertex():
    """Select uniformly at random a single vertex. Returns a tuple containing the coordinates of the vertex"""
    return (np.random.choice(range(GRID)), np.random.choice(range(GRID)))

def g(i, mag, d):
    """
    Returns g^i (mag) := exp(beta*mag[i])/sum_j(2cosh(beta*mag[j])).
    i should be the spin, *not* the index corresponding to that spin
    """
    numerator = exp(BETA * get_magnetisation(mag, i))
    denominator = sum([2*cosh(BETA * get_magnetisation(mag, j)) for j in range(1, CURRENT_D+1)])

    return numerator / denominator

def sample_new_spin_complete(current_spin, current_prop, d):

    # remove the current spin from the magnetisation
    adj_prop = current_prop - get_prop_update(current_spin) 
    adj_mag = magnetisation_from_propotion(adj_prop) / TOTAL_SPINS

    # compute the transition probabilities
    # only compute spins that we are currently using
    positive_spin_probs = [g(i, adj_mag, d) if i <= CURRENT_D else 0 for i in range(1, MAX_D+1)]
    negative_spin_probs = [g(-i, adj_mag, d) if i <= CURRENT_D else 0 for i in range(1, MAX_D+1)]

    conditional_measure = positive_spin_probs + negative_spin_probs
    
    # choose a new spin according to our conditional measure
    cdf = [0 for _ in conditional_measure]
    cdf[0] = conditional_measure[0]
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i-1] + conditional_measure[i]
    # zero out the un-used spins (this feels inefficient)
    for i in range(len(cdf)):
        if abs(get_spin(i)) > CURRENT_D: cdf[i] = 0

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    below = filter( 
        lambda i : unif <= cdf[i],
        list(range(2*MAX_D))
    )
    res = min(list(below))

    return get_spin(res)

def sample_new_spin_lattice(vertex, state, d):
    """
    Sample a new spin according to the conditional face-cubic measure on the square lattice. Does periodic boundary conditions by treating opposite edges as adjacent.

    vertex is a tuple containing the coordinates of the vertex to be updated
    """
    # compute adjacent magnetisation
    left_i = (vertex[0] - 1 % GRID, vertex[1] % GRID) 
    right_i = (vertex[0] + 1 % GRID, vertex[1] % GRID) 
    top_i = (vertex[0]  % GRID, vertex[1] + 1 % GRID) 
    bottom_i = (vertex[0]  % GRID, vertex[1] - 1 % GRID) 

    left = get_spin_vector(state[left_i])
    right = get_spin_vector(state[right_i])
    top = get_spin_vector(state[top_i])
    bottom = get_spin_vector(state[bottom_i])

    adj_mag = left + right + top + bottom

    # compute transition probabilities
    # only compute spins that we are currently using
    positive_spin_probs = [g(i, adj_mag, d) if i <= CURRENT_D else 0 for i in range(1, MAX_D+1)]
    negative_spin_probs = [g(-i, adj_mag, d) if i <= CURRENT_D else 0 for i in range(1, MAX_D+1)]

    conditional_measure = positive_spin_probs + negative_spin_probs

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    below = filter( 
        lambda i : unif <= cdf[i],
        list(range(2*MAX_D))
    )
    res = min(list(below))

    return get_spin(res)
    return

###################
#      PLOTS      #
###################

# Create a figure and axis
fig, ax = plt.subplots()

plt.style.use('_mpl-gallery-nogrid')

# make initial data
state = [[unif_ising() for _ in range(GRID)] for _ in range(GRID)]
state = np.array(state)

# plot grid
grid = ax.imshow(
    state, 
    origin='lower', 
    cmap=cmap_simple,
    norm=boundary_norm
    
)
# I don't want ticks
ax.set_xticks([])
ax.set_yticks([])

# Create a slider for temperature
ax_freq_slider = plt.axes([.9,.1,.01,.8])
freq_slider = Slider(ax_freq_slider, '$\\beta$', 0.0, 10.0, valinit=1, orientation='vertical')

# Update function for temperature
def update_slider(val):
    global BETA
    BETA = val
    fig.canvas.draw_idle()
#
# Connect the frequency slider to the update function
freq_slider.on_changed(update_slider)

def update(frame):
    # perform a Glauber update
    v = select_vertex()
    
    if CURRENT_GRAPH == GraphGeometry.COMPLETE:
        new_spin = sample_new_spin_complete(
            current_spin=state[v], 
            current_prop=proportions_from_state_unormalised(state), 
            d=CURRENT_D
        ) 
    elif CURRENT_GRAPH == GraphGeometry.LATTICE:
        pass
        # TODO
    state[v] = new_spin

    print(f"β =  {BETA},  d = {CURRENT_D}", end='\r')

    grid.set(data = state)

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=None,
    interval=INTERVAL,
    cache_frame_data=False,  # I just want to show this animation in a window, not save it
)

plt.show()
