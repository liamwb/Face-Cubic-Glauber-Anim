from enum import Enum
import numpy as np
from numpy import cosh, sinh, exp, log
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.cm as cm

from PIL import Image

import time
import argparse
import os

from numba import njit

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
LATTICE_GRID=212
COMPLETE_GRID=64

GRID = LATTICE_GRID
TOTAL_SPINS = GRID**2

K = 1  # single source of truth 
BETA = 1
MAX_K = 5  # I'm not sure how insightful going much higher than 4 is, and I'm not sure I could display the spins in a visually sensible way
MAX_BETA = 2*log(4)

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
    '#000000',  # -1: Black
    '#888888',  #  0: Grey (empty)
    '#FFFFFF',  # +1: White
]

# Create colormap
cmap_simple = ListedColormap(colors_simple)

# boundary norm so that my colours go exactly where I want
boundaries = [-1.5,-0.5,0.5,1.5]
boundary_norm = BoundaryNorm(boundaries, cmap_simple.N)

def unif_ising():
    return np.random.choice([-1,1])

def unif_spin():
    return np.random.choice([-1,0,1]) 

def proportions_from_state_unormalised(state):
    """Given a state, counts the number of spins of each kind. Returns a proportions vector of the form (-1,0,1)."""
    values, counts = np.unique(state, return_counts=True)

    prop = [0 for _ in range(3)]
    for count_index, spin in enumerate(values):
        prop[spin+1] = counts[count_index]
    
    return np.array(prop) 
    # Note that the simulation could be made more efficient by keeping track of the proportions vector with each update, instead of counting everything each step.

@njit
def magnetisation_from_propotion(proportion):
    return proportion[2]-proportion[0]


@njit
def get_prop_update(spin):
    """
    Given a spin, generates a vector with a 1 in the corresponding position. Intended use is to update proportions
    """
    res = np.zeros(3)
    res[spin+1] = 1
    return res

##############
# SIMULATION #
##############
@njit
def select_vertex(grid):
    """Select uniformly at random a single vertex. Returns a tuple containing the coordinates of the vertex"""
    return (np.random.randint(0,grid), np.random.randint(0,grid))

def p(s, adj_mag):
    """Returns the probability of updating to spin s given adjacent magnetisation adj_mag"""
    global K, BETA, TOTAL_SPINS
    denominator = exp(2*BETA*K*adj_mag)+ exp(- 2*BETA*K*adj_mag) + exp(BETA - BETA * K / TOTAL_SPINS)

    if s == -1:
        numerator = exp(-2*BETA*K*adj_mag)
    elif s == 1:
        numerator = exp(2*BETA*K*adj_mag)
    else: 
        numerator = exp(BETA - BETA * K / TOTAL_SPINS)

    return numerator / denominator

@njit
def p_num(s, adj_mag, beta, k, total_spins):
    """
    Returns the numerator of p(s, adj_mag).
    s should be one of -1, 0, 1
    """
    if s == -1:
        return exp(-2*beta*k*adj_mag)
    elif s == 1:
        return exp(2*beta*k*adj_mag)
    else:
        return exp(beta - beta * k / total_spins)

@njit
def p_denom(adj_mag, beta, k, total_spins):
    """
    Returns the denominator of p(s, adj_mag).
    """
    return exp(2*beta*k*adj_mag) + exp(-2*beta*k*adj_mag) + exp(beta - beta * k / total_spins)

@njit
def sample_new_spin_complete(current_spin, current_prop, beta, k, total_spins):

    # remove the current spin from the magnetisation
    adj_prop = current_prop - get_prop_update(current_spin) 
    adj_mag = magnetisation_from_propotion(adj_prop) / total_spins

    # compute the transition probabilities
    denominator = p_denom(adj_mag, beta, k, total_spins)
    conditional_measure = [p_num(s, adj_mag, beta, k, total_spins)/denominator for s in [-1,0,1]]

    # construct cdf
    cdf = np.cumsum(np.array(conditional_measure))

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    res=2  # If due to some fp weirdness unif > cdf[i], set to the last spin (+1)
    for i in range(3):
        if unif <= cdf[i]:
            res=i
            break

    return res - 1

@njit
def sample_new_spin_lattice(i, j, state, beta, k, total_spins):
    """
    Sample a new spin according to the conditional measure on the square lattice. Does periodic boundary conditions by treating opposite edges as adjacent.

    i and j are the coordinates of the vertex to be updated
    """
    grid = state.shape[0]
    # compute adjacent magnetisation  (% ==> periodic boundary conditions)
    left_i = ((i - 1) % grid, j % grid) 
    right_i = ((i + 1) % grid, j % grid) 
    top_i = (i  % grid, (j + 1) % grid) 
    bottom_i = (i  % grid, (j - 1) % grid) 

    left = state[left_i]
    right = state[right_i]
    top = state[top_i]
    bottom = state[bottom_i]

    adj_mag = left + right + top + bottom 

    # compute transition probabilities
    denominator = p_denom(adj_mag, beta, k, total_spins)
    conditional_measure = [p_num(s, adj_mag, beta, k, total_spins)/denominator for s in [-1,0,1]]

    # construct cdf
    cdf = np.cumsum(np.array(conditional_measure))

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    res=2  # If due to some fp weirdness unif > cdf[i], set to the last spin (+1)
    for i in range(3):
        if unif <= cdf[i]:
            res=i
            break

    return res - 1


def save_png_sequence(output_dir, num_frames=300, k=1, beta_init=1.0, geometry='lattice',
                       updates_per_frame=500):
    """Save a PNG sequence of just the grid (no UI elements) to the specified directory."""
    global K, BETA, CURRENT_GRAPH, GRID, TOTAL_SPINS, state

    K = k
    BETA = beta_init
    geometry_map = {'lattice': GraphGeometry.LATTICE, 'complete': GraphGeometry.COMPLETE}
    CURRENT_GRAPH = geometry_map[geometry]
    GRID = LATTICE_GRID if CURRENT_GRAPH == GraphGeometry.LATTICE else COMPLETE_GRID
    TOTAL_SPINS = GRID**2

    state = np.array([[unif_spin() for _ in range(GRID)] for _ in range(GRID)])

    os.makedirs(output_dir, exist_ok=True)

    for frame in range(num_frames):
        for _ in range(updates_per_frame):
            v = select_vertex(grid=GRID)
            if CURRENT_GRAPH == GraphGeometry.COMPLETE:
                new_spin = sample_new_spin_complete(
                    current_spin=state[v],
                    current_prop=proportions_from_state_unormalised(state),
                    beta=BETA,
                    k=K,
                    total_spins=TOTAL_SPINS
                )
            else:
                new_spin = sample_new_spin_lattice(
                    i=v[0],
                    j=v[1],
                    state=state,
                    beta=BETA,
                    k=K,
                    total_spins=TOTAL_SPINS
                )
            state[v] = new_spin

        rgba = cmap_simple(boundary_norm(state))          # (GRID, GRID, 4), floats 0..1
        img = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        img.save(os.path.join(output_dir, f'frame_{frame:04d}.png'))

        print('Saved frame {}/{}'.format(frame + 1, num_frames), end='\r')

    print('\nDone! {} frames saved to {}/'.format(num_frames, output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Glauber dynamics simulation')
    parser.add_argument('--save', type=str, default=None, help='Directory to save PNG sequence to')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to save')
    parser.add_argument('--K', type=int, default=1, help='Parameter K')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    parser.add_argument('--geometry', type=str, default='lattice', choices=['lattice', 'complete'],
                        help='Graph geometry')
    parser.add_argument('--updates-per-frame', type=int, default=500,
                        help='Glauber updates per animation frame')
    args = parser.parse_args()

    if args.save:
        save_png_sequence(args.save, args.frames, args.K, args.beta, args.geometry,
                          args.updates_per_frame)
    else:

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
            Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(0),lw=0),
            Rectangle([0,0],1,1,facecolor=scalar_map.to_rgba(-1),lw=0),
        ]
        def get_legend_labels(): return [
            '$+1$', '$0$', '$-1$' 
        ]
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
            valmax=MAX_BETA, 
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

        # Create a slider for K (vertical)
        d_ax = fig.add_axes([0.03,0.25,0.0225,0.63])
        k_slider = Slider(
            ax=d_ax, 
            label='$K$', 
            valmin=0, 
            valmax=MAX_K, 
            valinit=0, 
            orientation='vertical',
        )

        # Update function for K
        def update_d(val):
            global K, leg
            K = val

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
        k_slider.on_changed(update_d)

        # button to toggle GraphGeometry
        button_ax = fig.add_axes([0.375,0.9,0.25,0.07])
        button = Button(button_ax, "LATTICE") if CURRENT_GRAPH == GraphGeometry.LATTICE else Button(button_ax, "COMPLETE GRAPH")
        def toggle_geometry(event):
            global CURRENT_GRAPH, GRID, TOTAL_SPINS, button, grid, state
            if CURRENT_GRAPH == GraphGeometry.LATTICE:
                button.label.set_text("COMPLETE GRAPH")
                CURRENT_GRAPH = GraphGeometry.COMPLETE
                GRID = COMPLETE_GRID
            elif CURRENT_GRAPH == GraphGeometry.COMPLETE:
                button.label.set_text("SQUARE LATTICE")
                CURRENT_GRAPH = GraphGeometry.LATTICE
                GRID = LATTICE_GRID

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
            for _ in range(args.updates_per_frame):
                t1 = time.perf_counter()
                # perform a Glauber update
                v = select_vertex(grid=GRID)

                if CURRENT_GRAPH == GraphGeometry.COMPLETE:
                    new_spin = sample_new_spin_complete(
                        current_spin=state[v], 
                        current_prop=proportions_from_state_unormalised(state), 
                        beta=BETA,
                        k=K,
                        total_spins=TOTAL_SPINS
                    ) 
                elif CURRENT_GRAPH == GraphGeometry.LATTICE:
                    new_spin = sample_new_spin_lattice(
                        i=v[0], 
                        j=v[1], 
                        state=state, 
                        beta=BETA,
                        k=K,
                        total_spins=TOTAL_SPINS
                    )

                state[v] = new_spin

            t2 = time.perf_counter()

            print(f"β =  {BETA},  K = {K}, Frametime: {round((t2-t1)*1000, 3)}ms, S_t = {np.round(proportions_from_state_unormalised(state)/TOTAL_SPINS, decimals=2)}", end='            \r')

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
