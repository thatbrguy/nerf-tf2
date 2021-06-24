import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras.layers import Input, Concatenate

from nerf.utils import ray_utils

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

def get_nerf_model(model_name, num_units = 256):
    """
    Creates and returns a NeRF model.

    ## TODO: Verify consistency, indices, check for one off errors etc.
    ## Add comments to explain stuff!
    """
    assert model_name in ["coarse", "fine"]
    
    xyz = Input(shape = (3,), name = f"{model_name}/xyz")
    rays_d = Input(shape = (3,), name = f"{model_name}/rays_d")

    name = f"{model_name}/enc_xyz"
    enc_xyz = PositionalEncoder(L = 10, name = name)(xyz)

    name = f"{model_name}/enc_rays_d"
    enc_rays_d = PositionalEncoder(L = 4, name = name)(rays_d)

    value = enc_xyz

    for i in range(8):
        name = f"{model_name}/dense_{i}"
        value = Dense(num_units, name = name, activation = "relu")(value)

        if i == 4:
            name = f"{model_name}/concat_1"
            value = Concatenate(name = name)([value, enc_xyz])

    name =  f"{model_name}/sigma"
    sigma = Dense(1, activation = None, name = name)(value)

    name =  f"{model_name}/dense_8"
    bottleneck = Dense(num_units, activation = None, name = name)(value)
    
    name = f"{model_name}/concat_2"
    value = Concatenate(name = name)([bottleneck, enc_rays_d])
    
    name = f"{model_name}/dense_9"
    value = Dense(num_units // 2, name = name, activation = "relu")(value)

    name = f"{model_name}/rgb"
    rgb = Dense(3, activation = "sigmoid", name = name)(value)

    inputs = [xyz, rays_d]
    outputs = [rgb, sigma]

    model = Model(inputs = inputs, outputs = outputs)

    return model

class NeRF(Model):
    """
    Model to define custom fit, evaluate and predict operations.

    This class implements both coarse and fine models.
    """
    def __init__(self, params = None):

        super().__init__()
        self.params = params

        self.coarse_model = get_nerf_model(model_name = "coarse")
        self.fine_model = get_nerf_model(model_name = "fine")

    def train_step(self, data):
        """
        TODO: Docstring!
        """
        (rays_o, rays_d, near, far), (rgb,) = data

        # Getting data ready for the coarse model.
        (
            xyz_inputs, dir_inputs, 
            bin_data, t_vals_coarse,
        ) = ray_utils.create_input_batch_coarse_model(
            params = self.params, rays_o = rays_o, 
            rays_d = rays_d, near = near, far = far
        )

        # Performing a forward pass through the coarse model.
        with tf.GradientTape() as tape:
            rgb, sigma = self.coarse_model(xyz_inputs, dir_inputs)

            ## TODO: Handle loss.
            # coarse_loss == ???

        # Computing weights for each bin along each ray.
        weights = rays_utils.compute_bin_weights(
            bin_data, t_vals_coarse, sigma
        )

        # Getting data ready for the fine model.
        xyz_inputs, dir_inputs = ray_utils.create_input_batch_fine_model(
            params = self.params, rays_o = rays_o, 
            rays_d = rays_d, weights = weights, 
            t_vals_coarse = t_vals_coarse,
            bin_data = bin_data,
        )

        # Performing a forward pass through the fine model.
        with tf.GradientTape() as tape:
            rgb, sigma = self.fine_model(xyz_inputs, dir_inputs)

            ## TODO: Handle loss.
            # fine_loss == ???

        ## TODO: optimizer, grads, metrics etc.!
        pass

    def test_step(self, data):
        pass

    def predict_step(self, data):
        pass

class NeRFLite(Model):
    """
    Model to define custom fit, evaluate and predict operations.

    This class implements only the coarse model.
    """
    def __init__(self, params = None):
        
        super().__init__()
        self.params = params
        self.coarse_model = get_nerf_model(model_name = "coarse")

    def train_step(self, data):
        """
        TODO: Docstring
        """
        (rays_o, rays_d, near, far), (rgb,) = data

        # Getting data ready for the coarse model.
        (
            xyz_inputs, dir_inputs, 
            bin_data, t_vals_coarse,
        ) = ray_utils.create_input_batch_coarse_model(
            params = self.params, rays_o = rays_o, 
            rays_d = rays_d, near = near, far = far
        )

        # Performing a forward pass through the coarse model.
        with tf.GradientTape() as tape:
            rgb, sigma = self.coarse_model(xyz_inputs, dir_inputs)

            ## TODO: Handle loss.
            # coarse_loss == ???

        ## TODO: optimizer, grads, metrics etc.!
        pass

    def test_step(self, data):
        pass

    def predict_step(self, data):
        pass

if __name__ == '__main__':

    coarse_model = get_nerf_model(model_name = "coarse")
    fine_model = get_nerf_model(model_name = "fine")

    nerf = NeRF()
    nerf_lite = NeRFLite()
