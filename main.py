from enum import Enum
import numpy as np
from numpy import cosh, sinh, exp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.cm as cm

import time

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
# For the simulation on the complete graph, 64x64 seems about right
# For the simulation on the lattice graph, 64x64 seems about right
GRID = 212
TOTAL_SPINS = GRID**2

CURRENT_D = 1  # single source of truth 
BETA = 1
MAX_D = 5  # I'm not sure how insightful going much higher than 4 is, and I'm not sure I could display the spins in a visually sensible way
MAX_BETA = 4*CURRENT_D  # how far the beta slider should go

# 30 fps
INTERVAL = 33.33 

class GraphGeometry(Enum):
    COMPLETE = 1
    LATTICE = 2

CURRENT_GRAPH = GraphGeometry.LATTICE

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

def unif_spin():
    return np.random.choice([-1,1]) * np.random.choice([1,2,3,4,5][:CURRENT_D])

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
        res[MAX_D + (spin*-1) - 1] = 1
    return res

def get_spin_vector(spin):
    """
    Convertes a spin (represented by an integer) to the corresponding np vector
    """
    index = abs(spin)-1
    res = np.zeros(MAX_D)
    res[index] = np.sign(spin)
    return res


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
    # compute adjacent magnetisation  (% ==> periodic)
    left_i = ((vertex[0] - 1) % GRID, vertex[1] % GRID) 
    right_i = ((vertex[0] + 1) % GRID, vertex[1] % GRID) 
    top_i = (vertex[0]  % GRID, (vertex[1] + 1) % GRID) 
    bottom_i = (vertex[0]  % GRID, (vertex[1] - 1) % GRID) 

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

    # construct cdf
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

###################
#      PLOTS      #
###################

# Create a figure, axes
fig, ax = plt.subplots()

plt.style.use('_mpl-gallery-nogrid')

# make initial data
state = [[unif_spin() for _ in range(GRID)] for _ in range(GRID)]
state = np.array(state)

# plot grid
grid = ax.imshow(
    state, 
    origin='lower', 
    cmap=cmap_simple,
    norm=boundary_norm,
    animated=True
)

# legend for the spins
scalar_map = cm.ScalarMappable(norm=boundary_norm, cmap=cmap_simple)

def get_legend_elements(): return [
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(1),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-1),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(2),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-2),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(3),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-3),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(4),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-4),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(5),lw=0),
    Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-5),lw=0),
][:2*CURRENT_D]
def get_legend_labels(): return [
    '$e_1$', '$-e_1$', '$e_2$', '$-e_2$', '$e_3$', '$-e_3$', '$e_4$', '$-e_4$', '$e_5$', '$-e_5$', 
][:2*CURRENT_D]
legend_elements = get_legend_elements(); legend_labels = get_legend_labels();
leg = fig.legend(
    handles=legend_elements, 
    labels=legend_labels, 
    loc='upper right',
    fontsize='x-large'
)

# I don't want ticks
ax.set_xticks([])
ax.set_yticks([])

# adjust the main plot to make room for the sliders
# fig.subplots_adjust(right=0.75, bottom=0.25)
# fig.tight_layout()

# Create a slider for temperature (horizontal)
beta_ax = fig.add_axes([0.04,0.04,0.21,0.03])
beta_slider = Slider(
    ax=beta_ax, 
    label='$\\beta$', 
    valmin=0.0, 
    valmax=10.0, 
    valinit=0, 
    orientation='horizontal',
facecolor='black')

# Update function for temperature
def update_beta(val):
    global BETA
    BETA = val
    fig.canvas.draw_idle()

    # put the new value in beta_box
    beta_box.set_val(round(val, 3))

# Connect the temperature slider to the update function
beta_slider.on_changed(update_beta)
# don't show the valtext because we have a textbox for beta
beta_slider.valtext.set_visible(False)

# Create a slider for dimension (vertical)
d_ax = fig.add_axes([0.03,0.25,0.0225,0.63])
d_slider = Slider(
    ax=d_ax, 
    label='$d$', 
    valmin=1, 
    valmax=5, 
    valinit=1, 
    orientation='vertical',
    valstep=1
)

# Update function for dimension
def update_d(val):
    global CURRENT_D, leg
    CURRENT_D = val

    # update the legend
    # this is a hack -- we make the old legend invisible and then de-reference it
    # hopefully the old legend gets garbage collected...
    leg.set(visible=False)
    legend_elements = get_legend_elements(); legend_labels = get_legend_labels();
    leg = fig.legend(
        handles=legend_elements, 
        labels=legend_labels, 
        loc='upper right'
    )

    fig.canvas.draw_idle()

# Connect the temperature slider to the update function
d_slider.on_changed(update_d)

# button to toggle GraphGeometry
button_ax = fig.add_axes([0.375,0.9,0.25,0.07])
button = Button(button_ax, "LATTICE") if CURRENT_GRAPH == GraphGeometry.LATTICE else Button(button_ax, "COMPLETE GRAPH")
def toggle_geometry(event):
    global CURRENT_GRAPH, GRID, TOTAL_SPINS, button, grid, state
    if CURRENT_GRAPH == GraphGeometry.LATTICE:
        button.label.set_text("COMPLETE GRAPH")
        CURRENT_GRAPH = GraphGeometry.COMPLETE
        GRID = 64
    elif CURRENT_GRAPH == GraphGeometry.COMPLETE:
        button.label.set_text("SQUARE LATTICE")
        CURRENT_GRAPH = GraphGeometry.LATTICE
        GRID = 212

    # re-compute normalising constant
    TOTAL_SPINS = GRID**2

    # re-generate a uniform state
    state = [[unif_spin() for _ in range(GRID)] for _ in range(GRID)]
    state = np.array(state)

    # remove and replace grid
    grid.remove()
    grid = ax.imshow(
        state, 
        origin='lower', 
        cmap=cmap_simple,
        norm=boundary_norm,
        animated=True
    )


button.on_clicked(toggle_geometry)

# entry fields for beta slider
beta_box_ax = fig.add_axes([0.1,0.1,0.1,0.075])
beta_box = TextBox(beta_box_ax, "$\\beta$")
def submit_beta(expr):
    global BETA
    try:
        val = float(expr)
        BETA = val
        beta_slider.set_val(val)
    except:  # if there's some junk in the input
        beta_box.set_val("")

beta_box.on_submit(submit_beta)
beta_box.set_val(0)


# main loop for the animation
def update(frame, *fargs):
    for _ in range(500):
        t1 = time.perf_counter()
        # perform a Glauber update
        v = select_vertex()

        if CURRENT_GRAPH == GraphGeometry.COMPLETE:
            new_spin = sample_new_spin_complete(
                current_spin=state[v], 
                current_prop=proportions_from_state_unormalised(state), 
                d=CURRENT_D
            ) 
        elif CURRENT_GRAPH == GraphGeometry.LATTICE:
            new_spin = sample_new_spin_lattice(
                vertex=v, 
                state=state, 
                d=CURRENT_D
            )

        state[v] = new_spin

    t2 = time.perf_counter()

    print(f"β =  {BETA},  d = {CURRENT_D}, Frametime: {round((t2-t1)*1000, 3)}ms", end='\r')

    grid.set_data(state)

    # if frame >= 5:
    #     exit()

    return fargs  # pass the artists back for blitting

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=None,
    interval=INTERVAL,
    cache_frame_data=False,  # I just want to show this animation in a window, not save it
    blit=True  
)

plt.show()
