import tensorflow as tf
import numpy as np
import libtcg_utils as tcg_utils
import os
def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image
root=