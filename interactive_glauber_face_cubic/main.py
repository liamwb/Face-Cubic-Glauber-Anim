from enum import Enum
import numpy as np
from numpy import cosh, sinh, exp
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
LATTICE_GRID=400
COMPLETE_GRID=128

GRID = LATTICE_GRID
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
    '#9FE58E',  # -5: Soft sage
    '#E58EC2',  # -4: Soft pink
    '#9F8EE5',  # -3: Soft lavender
    '#8EB1E5',  # -2: Soft periwinkle
    '#FFFFFF',
    # '#8EE5E5',  # -1: Soft cyan
    '#FFF000',  #  0: Black (unused)
    # '#E58E8E',  #  1: Soft rose
    '#000000',
    '#E5C28E',  #  2: Soft peach
    '#D4E58E',  #  3: Soft lime
    '#8EE5B1',  #  4: Soft mint
    '#D48EE5',  #  5: Soft lilac
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

@njit
def magnetisation_from_propotion(proportion):
    return np.array([proportion[i] - proportion[i+MAX_D] for i in range(MAX_D)])

@njit
def get_prop_index(spin):
    """
    Given a spin -MAX_D, ..., -1, 1, ..., MAX_D, returns the index of a proportions vector that corresponds.
    Recall proportions vectors are of the form (#e1, #e2, ..., #-e1, ... #-eMAXD).
    """
    if spin > 0 :  # just need to index from zero
        return spin - 1
    else:  # need to start from MAX_D
        return MAX_D + (- spin) - 1

@njit
def get_spin(prop_index):
    """
    Given a proportions index, return the corresponding spin
    """
    if prop_index < MAX_D:  # spin is positive
        return prop_index + 1
    else:  # spin is negative
        return MAX_D - prop_index - 1

@njit
def get_magnetisation(mag, spin):
    """
    Given a magnetisation vector and a spin, returns the magnetisation in the corresponding direction
    """
    if spin > 0:  # just need to index from zero
        return mag[spin-1]
    else:  # if the spin is negative, return whatever is in the vector *-1
        return - mag[abs(spin)-1]

@njit
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
@njit
def select_vertex(grid):
    """Select uniformly at random a single vertex. Returns a tuple containing the coordinates of the vertex"""
    return (np.random.randint(0,grid), np.random.randint(0,grid))

def g(i, mag, d):
    """
    Returns g^i (mag) := exp(beta*mag[i])/sum_j(2cosh(beta*mag[j])).
    i should be the spin, *not* the index corresponding to that spin
    """
    numerator = exp(BETA * get_magnetisation(mag, i))
    denominator = sum([2*cosh(BETA * get_magnetisation(mag, j)) for j in range(1, d+1)])
    return numerator / denominator

@njit
def g_num(i, mag, beta):
    """
    Returns the numerator of g^i (mag) := exp(beta*mag[i])/sum_j(2cosh(beta*mag[j])).
    i should be the spin, *not* the index corresponding to that spin
    """
    return exp(beta * get_magnetisation(mag, i))

@njit
def g_denom(mag, beta, d):
    """
    Returns the denominator of g^i (mag) := exp(beta*mag[i])/sum_j(2cosh(beta*mag[j])).
    i should be the spin, *not* the index corresponding to that spin
    """
    return sum([2*cosh(beta * get_magnetisation(mag, j)) for j in range(1, d+1)])

@njit
def sample_new_spin_complete(current_spin, current_prop, beta, d, total_spins):

    # remove the current spin from the magnetisation
    adj_prop = current_prop - get_prop_update(current_spin) 
    adj_mag = magnetisation_from_propotion(adj_prop) / total_spins

    # compute the transition probabilities
    # only compute spins that we are currently using
    denominator = g_denom(adj_mag, beta, d)
    positive_spin_probs = [g_num(i, adj_mag, beta)/denominator if i <= d else 0 for i in range(1, MAX_D+1)]
    negative_spin_probs = [g_num(-i, adj_mag, beta)/denominator if i <= d else 0 for i in range(1, MAX_D+1)]

    conditional_measure = positive_spin_probs + negative_spin_probs

    # construct cdf
    cdf = np.cumsum(np.array(conditional_measure))

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    res=d  # If due to some fp weirdness unif > cdf[i], set to the last spin
    for i in range(2*MAX_D):
        if unif <= cdf[i]:
            res=i
            break

    return get_spin(res)

@njit
def sample_new_spin_lattice(i, j, state, beta, d):
    """
    Sample a new spin according to the conditional face-cubic measure on the square lattice. Does periodic boundary conditions by treating opposite edges as adjacent.

    i and j are the the coordinates of the vertex to be updated
    """
    grid=state.shape[0]
    # compute adjacent magnetisation  (% ==> periodic boundary conditions)
    left_i = ((i - 1) % grid, j % grid) 
    right_i = ((i + 1) % grid, j % grid) 
    top_i = (i  % grid, (j + 1) % grid) 
    bottom_i = (i  % grid, (j - 1) % grid) 

    left = state[left_i]
    right = state[right_i]
    top = state[top_i]
    bottom = state[bottom_i]

    # compute adjacent magnetisation
    adj_mag = np.zeros(MAX_D)

    for spin in [left, right, top, bottom]:
        adj_mag[abs(spin)-1] += 1 if spin > 0 else -1

    # compute transition probabilities
    # only compute spins that we are currently using
    denominator = g_denom(adj_mag, beta, d)
    positive_spin_probs = [g_num(i, adj_mag, beta)/denominator if i <= d else 0 for i in range(1, MAX_D+1)]
    negative_spin_probs = [g_num(-i, adj_mag, beta)/denominator if i <= d else 0 for i in range(1, MAX_D+1)]

    conditional_measure = positive_spin_probs + negative_spin_probs

    # construct cdf
    cdf = np.cumsum(np.array(conditional_measure))

    # sample via unif(0,1) noise
    unif = np.random.uniform(0,1)
    # choose the largest index where the cdf is still bigger than the noise
    res=d  # If due to some fp weirdness unif > cdf[i], set to the last spin
    for i in range(2*MAX_D):
        if unif <= cdf[i]:
            res=i
            break

    return get_spin(res)


def save_png_sequence(output_dir, num_frames=300, 
                      d_init=1, 
                      beta_init=1.0, 
                      geometry='lattice',
                      updates_per_frame=500):
    """Save a PNG sequence of just the grid (no UI elements) to the specified directory."""
    global CURRENT_D, BETA, CURRENT_GRAPH, GRID, TOTAL_SPINS, state

    CURRENT_D = d_init
    BETA = beta_init
    geometry_map = {'lattice': GraphGeometry.LATTICE, 'complete': GraphGeometry.COMPLETE}
    CURRENT_GRAPH = geometry_map[geometry]
    GRID = LATTICE_GRID if CURRENT_GRAPH == GraphGeometry.LATTICE else COMPLETE_GRID
    TOTAL_SPINS = GRID**2

    state = np.array([[unif_spin() for _ in range(GRID)] for _ in range(GRID)])

    # fig_save, ax_save = plt.subplots()
    # ax_save.axis('off')
    # fig_save.subplots_adjust(0, 0, 1, 1)
    #
    # grid_save = ax_save.imshow(state, origin='lower', cmap=cmap_simple, norm=boundary_norm)

    os.makedirs(output_dir, exist_ok=True)

    for frame in range(num_frames):
        for _ in range(updates_per_frame):
            v = select_vertex(grid=GRID)
            if CURRENT_GRAPH == GraphGeometry.COMPLETE:
                new_spin = sample_new_spin_complete(
                    current_spin=state[v],
                    current_prop=proportions_from_state_unormalised(state),
                    beta=BETA,
                    d=CURRENT_D,
                    total_spins=TOTAL_SPINS
                )
            else:
                new_spin = sample_new_spin_lattice(
                    i=v[0],
                    j=v[1],
                    state=state,
                    beta=BETA,
                    d=CURRENT_D
                )
            state[v] = new_spin

        rgba = cmap_simple(boundary_norm(state))          # (GRID, GRID, 4), floats 0..1
        img = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        img.save(os.path.join(output_dir, f'frame_{frame:04d}.png'))

        print('Saved frame {}/{}'.format(frame + 1, num_frames), end='\r')

    # plt.close(fig_save)
    print('\nDone! {} frames saved to {}/'.format(num_frames, output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Glauber dynamics simulation')
    parser.add_argument('--save', type=str, default=None, help='Directory to save PNG sequence to')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to save')
    parser.add_argument('--d', type=int, default=1, help='Spin dimension (1-5)')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    parser.add_argument('--geometry', type=str, default='lattice', choices=['lattice', 'complete'],
                        help='Graph geometry')
    parser.add_argument('--updates-per-frame', type=int, default=500,
                        help='Glauber updates per animation frame')
    parser.add_argument('--grid', type=int, default=400, help='Size of the square lattice')
    parser.add_argument('--complete', type=int, default=120, help='Size of the complete graph')
    args = parser.parse_args()

    LATTICE_GRID= args.grid
    COMPLETE_GRID= args.complete
    GRID=LATTICE_GRID
    TOTAL_SPINS = GRID**2

    if args.save:
        save_png_sequence(args.save, args.frames, args.d, args.beta, args.geometry,
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
            animated=True,
            interpolation = "none"
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
                animated=True,
                interpolation = "none"
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
                        d=CURRENT_D,
                        total_spins=TOTAL_SPINS
                    ) 
                elif CURRENT_GRAPH == GraphGeometry.LATTICE:
                    new_spin = sample_new_spin_lattice(
                        i=v[0], 
                        j=v[1], 
                        state=state, 
                        beta=BETA,
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
