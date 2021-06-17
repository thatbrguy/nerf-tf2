# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

from nerf.utils import pose_utils

def get_rays():
    """
    Gets ray origin and ray directions in the world coordinate system.

    TODO: Elaborate.
    """
    H_vals = np.arange(H, dtype = np.float64)
    W_vals = np.arange(W, dtype = np.float64)

    x, y = np.meshgrid(W_vals, H_vals, indexing = "xy")

    ## TODO: Write the more accurate version of this as well! 
    ## Think of using cx, cy, fx, fy maybe?
    x_vals = x - (W / 2)
    y_vals = y - (H / 2)
    z_vals = np.full(x_vals.shape, focal, dtype = np.float64)

    directions = np.stack([x_vals, y_vals, z_vals], axis = -1)

    ## TODO: Check output for correctness!
    rays_d = pose_utils.rotate_vectors(c2w, directions)
    rays_o = np.broadcast_to(c2w[:, 3], rays_d.shape)

    ## (TODO: Shape) --> (H*W, 3)
    rays_d = rays_d.reshape(-1, 3)
    rays_o = rays_o.reshape(-1, 3)
    
    return rays_o, rays_d

def create_input_batch_coarse_model(params, rays_o, rays_d):
    """
    Given rays_o and rays_d, creates a batch of inputs that 
    can be fed to the coarse model.

    TODO: Elaborate!
    """
    pass

def create_input_batch_fine_model(params, *args, **kwargs):
    """
    Creates batch of inputs for the fine model.

    TODO: Elaborate!
    """
    pass

if __name__ == '__main__':

    pass
