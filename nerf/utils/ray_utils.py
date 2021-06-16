# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

from nerf.utils import pose_utils

def get_rays_tf(H, W, focal, c2w):
    """
    Gets ray origin and ray directions, TensorFlow.

    TODO: Elaborate.
    """
    W_vals = tf.range(W, dtype = tf.float32)
    H_vals = tf.range(H, dtype = tf.float32)

    x, y = tf.meshgrid(W_vals, H_vals, indexing = "xy")

    x_vals = x - (W / 2)
    y_vals = y - (H / 2)
    z_vals = tf.fill(x_vals.shape, focal, dtype = tf.float32)

    directions = tf.stack([x_vals, y_vals, z_vals], axis = -1)

    ## TODO: Separate rotation function for tensorflow?
    rays_d = pose_utils.rotate_vectors(c2w, directions)
    ## rays_o = tf.broadcast_to something! TODO: Finish!
    
    return rays_o, rays_d

def get_rays_np():
    """
    Gets ray origin and ray directions, NumPy.

    TODO: Elaborate.
    """
    H_vals = np.arange(H, dtype = np.float64)
    W_vals = np.arange(W, dtype = np.float64)

    x, y = np.meshgrid(W_vals, H_vals, indexing = "xy")

    x_vals = x - (W / 2)
    y_vals = y - (H / 2)
    z_vals = np.full(x_vals.shape, focal, dtype = np.float64)

    directions = np.stack([x_vals, y_vals, z_vals], axis = -1)

    rays_d = pose_utils.rotate_vectors(c2w, directions)
    ## rays_o = np.broadcast_to something! TODO: Finish!
    
    return rays_o, rays_d

if __name__ == '__main__':

    pass
