import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

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
        output = tf.reshape(intermediate, (x.shape[0], -1))

        return output

class NeRF(Model):
    """
    TODO: Docstring
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        pass

if __name__ == '__main__':

    PE = PositionalEncoder(2)
