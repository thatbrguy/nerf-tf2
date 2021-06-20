# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

from nerf.utils import pose_utils

def get_rays(H, W, focal, c2w):
    """
    Gets ray origin and ray directions in the world coordinate system.

    TODO: Elaborate.

    NOTE, TODO, IMPORTANT: Since we are using only one focal length, 
    should we make sure that H == W?
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
    ## (H, W, 3) --> (H*W, 3) TODO: Verify
    directions = directions.reshape(-1, 3)

    ## TODO: Check output for correctness!
    rays_d = pose_utils.rotate_vectors(c2w, directions)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    
    return rays_o, rays_d

def create_input_batch_coarse_model(params, rays_o, rays_d, near, far):
    """
    Given rays_o and rays_d, creates a batch of inputs that 
    can be fed to the coarse model.

    Args:
        rays_o  : TODO (type, explain) with shape (N_rays, 3).
        rays_d  : TODO (type, explain) with shape (N_rays, 3).
        near    : TODO (type, explain) with shape (N_rays, 1).
        far     : TODO (type, explain) with shape (N_rays, 1).

    Returns:
        xyz_inputs      : TODO: Explain
        rays_d_inputs   : TODO: Explain

    TODO: Elaborate!
    """
    if params.sampling.lindisp:
        raise NotImplementedError()

    # Getting N_coarse+1 evenly spaced samples between near and far values 
    # for each ray. Shape of vals --> (N_rays, N_coarse + 1, 1)
    vals = tf.linspace(near, far, num = params.sampling.N_coarse + 1, axis = 1)
    
    # TODO: Explain bin_edges
    # Shape of bin_edges --> (N_rays, N_coarse + 1)
    bin_edges = tf.squeeze(vals, axis = 2)

    #################################################
    ## Stratified Sampling.
    #################################################

    # Shape of lower_edges --> (N_rays, N_coarse)
    lower_edges = bin_edges[:, :-1]
    # Shape of upper_edges --> (N_rays, N_coarse)
    upper_edges = bin_edges[:, 1:]
    
    if params.sampling.perturb:
        # Shape of bin_widths --> (N_rays, N_coarse)
        bin_widths = upper_edges - lower_edges

        # Getting random uniform samples in the range [0, 1). 
        # Shape of u_vals --> (N_rays, N_coarse)
        u_vals = tf.random.uniform(shape = bin_widths.shape)

        # Using the below logic, we get one random sample in each bin.
        # Shape of t_vals --> (N_rays, N_coarse)
        t_vals = lower_edges + (u_vals * bin_widths)

    elif not params.sampling.perturb:

        # Getting the mid point of each bin.
        # Shape of bin_mids --> (N_rays, N_coarse)
        bin_mids = 0.5 * (lower_edges + upper_edges)

        # In this case, we directly use bin_mids as t_vals.
        # Shape of t_vals --> (N_rays, N_coarse)
        t_vals = bin_mids

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz --> (N_rays, N_samples, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    # Shape of rays_d_broadcasted --> (N_rays, N_samples, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d, xyz.shape)
    
    # Shape of rays_d_inputs --> (N_rays * N_samples, 3)
    rays_d_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    # Shape of xyz_inputs --> (N_rays * N_samples, 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))
    
    return xyz_inputs, rays_d_inputs

def create_input_batch_fine_model(params, *args, **kwargs):
    """
    Creates batch of inputs for the fine model.

    TODO: Elaborate!
    """
    pass

if __name__ == '__main__':

    # NOTE, TODO, IMPORTANT: Since we are using only one focal length, 
    # should we make sure that H == W?
    rays_o, rays_d = get_rays(H = 500, W = 500, focal = 250, c2w = np.eye(4))
    pass
