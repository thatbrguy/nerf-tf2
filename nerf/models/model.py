# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras.layers import Input, Concatenate

class PositionalEncoder(Layer):
    """
    TODO: Docstring and verify code!
    """
    def __init__(self, L, name = None):
        """
        TODO: Docstring
        """
        super().__init__(trainable = False, name = name)

        self.L = L
        exponents = tf.range(self.L, dtype = tf.float32)
        self.multipliers = (2 ** exponents) * np.pi
        self.multipliers =  tf.reshape(self.multipliers, (1, 1, -1))

    def call(self, x):
        """
        TODO: Docstring
        """
        expanded = tf.expand_dims(x, axis = -1) * self.multipliers
        
        cos_ = tf.math.cos(expanded)
        sin_ = tf.math.sin(expanded)

        intermediate = tf.stack([sin_, cos_], axis = -1)

        ## TODO: Check if using Flatten is correct.
        # output = tf.reshape(intermediate, (x.shape[0], -1))
        output = Flatten()(intermediate)

        return output

def get_nerf_model(num_units = 256):
    """
    Creates and returns a NeRF model.
    """
    rays_o = Input(shape = (3,), name = "pos_vec")
    rays_d = Input(shape = (3,), name = "dir_vec")

    enc_rays_o = PositionalEncoder(L = 10, name = "enc_rays_o")(rays_o)
    enc_rays_d = PositionalEncoder(L = 4, name = "enc_rays_d")(rays_d)

    value = enc_rays_o

    for i in range(8):
        value = Dense(num_units, activation = "relu")(value)

        if i == 4:
            value = Concatenate()([value, enc_rays_o])

    sigma = Dense(1, activation = None, name = "sigma")(value)
    bottleneck = Dense(num_units, activation = None)(value)
    
    value = Concatenate()([bottleneck, enc_rays_d])
    value = Dense(num_units // 2, activation = "relu")(value)
    rgb = Dense(3, activation = "sigmoid", name = "rgb")(value)

    inputs = [rays_o, rays_d]
    outputs = [rgb, sigma]

    model = Model(inputs = inputs, outputs = outputs)

    return model

if __name__ == '__main__':

    model = get_nerf_model()

