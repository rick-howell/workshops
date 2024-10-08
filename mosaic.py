import numpy as np
import cv2
import os

DOWNSCALE_FACTOR = 2
GRID_UNIT = 12     # The dimension of each tile in the grid

# Load the image
path = 'path/to/image.xyz'

if not os.path.isfile(path):
    raise FileNotFoundError(f'File {path} not found')


# Part 1: Section off the image into a grid --------------------------- #

img = cv2.imread(path)
dims = img.shape
img = cv2.resize(img, (dims[1] // DOWNSCALE_FACTOR, dims[0] // DOWNSCALE_FACTOR))

h, w, _ = img.shape     # img.shape returns the height, width, and number of channels of the image
n_rows = h // GRID_UNIT     # Number of rows in the grid
n_cols = w // GRID_UNIT     # Number of columns in the grid



# ------------------------------------------------------------------------ #

def average(partition: np.ndarray) -> float:
    # TODO: Calculate the average color of the partition

    # Get the mean over the axis (0, 1)

    return 0.0


def add_noise(color: np.ndarray) -> np.ndarray:
    # TODO: Add noise to the color
    
    # Generate Noise

    # Add the noise to the color

    return color


# ------------------------------------------------------------------------ #

new_img = np.zeros((h, w, 3), dtype=np.uint8)

for i in range(n_rows):
    for j in range(n_cols):
        # Get the partition
        row_start = i * GRID_UNIT
        row_end = (i + 1) * GRID_UNIT

        col_start = j * GRID_UNIT
        col_end = (j + 1) * GRID_UNIT

        partition = img[row_start:row_end, col_start:col_end]


        # TODO: Perform the mosaic effect
        # Average the color of the partition

        # Add noise to the color


        # TODO: Clip the values to be between 0 and 255
        # Make sure the values are in the correct format!


        # TODO: Set the partition to the average color
        # Assign the color to the partition



# Part 2: Recolor the image ------------------------------------------- #

import colorsys

# The gradient should be a 256 x 3 array
# We'll use HLS color space: Hue, Lightness, Saturation
g_i = (0.5, 0.1, 0.5)     # Initial gradient color
g_f = (0.2, 0.9, 1.0)     # Final gradient color

def make_gradient(g_i: tuple, g_f: tuple) -> np.ndarray:

    # The gradient will be a 256 x 3 array
    gradient = np.zeros((256, 3), dtype=np.float32)

    for i in range(256):

        c1 = 0.0
        c2 = 0.0
        c3 = 0.0


        # ----------------------------------------- #

        # TODO: Interpolate the gradient color




        # ----------------------------------------- #

        if 0 <= c1 <= 1 and 0 <= c2 <= 1 and 0 <= c3 <= 1:
            pass
        else:
            raise ValueError('Color Channel values must be between 0 and 1')

        # Convert to RGB
        r, g, b = colorsys.hls_to_rgb(c1, c2, c3)
        r, g, b = np.fmod([r, g, b], 1.0) * 255

        # For cv2, we need to convert to BGR
        gradient[i] = [b, g, r]
        gradient = gradient.astype(np.uint8)

    
    return gradient


# Make the gradient
gradient = make_gradient(g_i, g_f)


def recolor(image, gradient) -> np.ndarray:
    # The second way we can recolor is by making a gradient over the whole image space
    # and then using overlay to recolor the image
    h, w, _ = image.shape
    new_img = np.zeros((h, w, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # We will then use a colormap to get the gradient position from the grayscale image

    '''
    TODO
    For each pixel in the image, get the gradient position from the grayscale image
    Then, color the new image with the gradient color
    '''

    return new_img



new_img = recolor(new_img, gradient)

# Lastly, we'll add grid lines to the image

lc = 64
line_color = (lc, lc, lc)

for i in range(n_rows):
    cv2.line(new_img, (0, i*GRID_UNIT), (w, i*GRID_UNIT), line_color, 1)

for i in range(n_cols):
    cv2.line(new_img, (i*GRID_UNIT, 0), (i*GRID_UNIT, h), line_color, 1)



# Display the image ---------------------------------------------------- #

cv2.imshow('Image', new_img)
cv2.waitKey(0)