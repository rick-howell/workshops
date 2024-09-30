# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
n_particles = 2048
resolution = 1024  # pixels per side
field_size = 10  # field size in units
force_scale = 0.0125  # scale of the vector field
decay_rate = 0.95  # decay rate of the image

# TODO: Initialize particle positions and velocities

# ================================= #

# ================================= #

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-field_size/2, field_size/2)
ax.set_ylim(-field_size/2, field_size/2)

# Create an image to store the particle positions and trails
image = np.zeros((resolution, resolution))

# TODO: Choose a Colormap!
# Colormaps can be found here: https://matplotlib.org/stable/gallery/color/colormap_reference.html
'''
supported values are: 
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
'''

# ================================= #
colormap = 'bone'
# ================================= #


# Create image plot
img_plot = ax.imshow(image, extent=[-field_size/2, field_size/2, -field_size/2, field_size/2], 
                     interpolation='nearest', vmin=0, vmax=1, cmap=colormap)

def vector_field(X, Y):
    """
    Define a vector field F(x, y) = U(x, y)i + V(x, y)j for the particles to follow
    X, Y: the x and y coordinates of the vector field
    """
    # TODO: Implement the vector field function
    pass

def random_respawn(rate=0.01):
    """
    Respawn particles randomly with a given rate
    rate: the rate at which particles should be respawned
    """
    global particles, velocities
    # TODO: Implement random respawn function


    pass

def index_respawn(particle_index):
    """
    Respawn particles at a given index
    particle_index: boolean array indicating which particles to respawn
    """
    global particles, velocities
    # TODO: Implement index-based respawn function

    
    pass

def update(frame):
    """
    Update the particle positions and image
    frame: the current frame number

    Returns: a tuple containing the updated image plot
    """

    global particles, velocities, image

    # TODO: Implement the update function

    # ================================= #
    

    # ================================= #

    # Update image plot
    # Can use either of the following methods to update the image plot
    # img_plot.set_array(image)
    img_plot.set_data(image)

    return img_plot,  # Note the comma here, as FuncAnimation expects an iterable


fig.tight_layout()
# Create animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()