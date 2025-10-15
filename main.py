import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
GRID = 128
D = 1
BETA = 1


####################
# HELPER FUNCTIONS #
####################

def unif_ising():
    return np.random.choice([1,2])

def max_beta(): return (4*d)

###################
#      PLOTS      #
###################

# Create a figure and axis
fig, ax = plt.subplots()

plt.style.use('_mpl-gallery-nogrid')

# make data
state = [[unif_ising() for _ in range(GRID)] for _ in range(GRID)]
state = np.array(state)

# plot grid
ax.imshow(state, origin='lower')

# Create a slider for temperature
ax_freq_slider = plt.axes([.9,.1,.01,.8])
freq_slider = Slider(ax_freq_slider, '$\\beta$', 0.1, 10.0, valinit=1, orientation='vertical')

# Update function for temperature
def update_slider(val):
    BETA = val
    fig.canvas.draw_idle()
#
# Connect the frequency slider to the update function
freq_slider.on_changed(update_slider)

##############
# SIMULATION #
##############
def select_vertex():
    """Select uniformly at random a single vertex"""
    return (np.random.choice(range(GRID)), np.random.choice(range(GRID)))

def update():
    v = select_vertex()
    new_spin = unif_ising()
    state[v] = new_spin

ani = animation.FuncAnimation(
    fig, update, 1000
)


plt.show()
