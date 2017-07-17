"""
Process an image to pass to our CNN.
"""
# !/usr/bin/env python
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def process_image(image, target_shape):
    """
    Turn image to an array
    params:
    image = the image
    target_shape = the target shape in height, width, depth
    output:
    im_arr
    """
    # Load the image.
    height, width, _ = target_shape
    image = load_img(image, target_size=(height, width))

    # Turn the image into an array and normalize.
    img_arr1 = img_to_array(image)
    im_arr2 = (img_arr1 / 255.).astype(np.float32)

    return im_arr2