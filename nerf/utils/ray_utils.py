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

    TODO: Support using fu, fv, cu, cv (get intrinsic matrix 
    as argument) and H != W.
    """
    assert H == W, (
        "Currently this function is written assuming "
        "H == W and fu == fv. In a later version, "
        "images with H != W and intrinsic matrices "
        "with fu, fv, cu, cv will be supported."
    ) 

    H_vals = np.arange(H, dtype = np.float64)
    W_vals = np.arange(W, dtype = np.float64)

    x, y = np.meshgrid(W_vals, H_vals, indexing = "xy")

    x_vals = x - (W / 2)
    y_vals = y - (H / 2)
    z_vals = np.full(x_vals.shape, focal, dtype = np.float64)

    directions = np.stack([x_vals, y_vals, z_vals], axis = -1)
    # (H, W, 3) --> (H*W, 3) TODO: Verify
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
        dir_inputs   : TODO: Explain

    TODO: Elaborate!

    TODO: Implement lindisp.
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

    # Shape of left_edges --> (N_rays, N_coarse)
    left_edges = bin_edges[:, :-1]
    # Shape of right_edges --> (N_rays, N_coarse)
    right_edges = bin_edges[:, 1:]
    
    if params.sampling.perturb:
        # Shape of bin_widths --> (N_rays, N_coarse)
        bin_widths = right_edges - left_edges

        # Getting random uniform samples in the range [0, 1). 
        # Shape of u_vals --> (N_rays, N_coarse)
        u_vals = tf.random.uniform(shape = bin_widths.shape)

        # Using the below logic, we get one random sample in each bin.
        # Shape of t_vals --> (N_rays, N_coarse)
        t_vals = left_edges + (u_vals * bin_widths)

    elif not params.sampling.perturb:

        # Getting the mid point of each bin.
        # Shape of bin_mids --> (N_rays, N_coarse)
        bin_mids = 0.5 * (left_edges + right_edges)

        # In this case, we directly use bin_mids as t_vals.
        # Shape of t_vals --> (N_rays, N_coarse)
        t_vals = bin_mids

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz --> (N_rays, N_coarse, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    # Shape of rays_d_broadcasted --> (N_rays, N_coarse, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d, xyz.shape)
    
    # Shape of dir_inputs --> (N_rays * N_coarse, 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    # Shape of xyz_inputs --> (N_rays * N_coarse, 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))

    # Collecting bin_data for returning.
    bin_data = {
        "bin_edges": bin_edges, "left_edges": left_edges,
        "right_edges": right_edges, "bin_widths": bin_widths,
    }
    
    return xyz_inputs, dir_inputs, bin_data, t_vals

def create_input_batch_fine_model(params, rays_o, rays_d, weights, bin_data, t_vals_coarse):
    """
    Creates batch of inputs for the fine model.

    TODO: Elaborate!

    TODO: Consider explaining concepts briefly over certain 
    lines. Detailed description can be provided elsewhere.
    
    TODO: Handle weights calculation. Assuming the weights will be stored 
    in a variable called weights. 
    
    Shape of weights must be --> (N_rays, N_coarse)
    """
    # Extracting useful content from bin_data
    left_edges = bin_data["left_edges"]
    bin_widths = bin_data["bin_widths"]

    # Creating pdf from weights.
    # Shape of weights --> (N_rays, N_coarse)
    weights = weights + 1e-5 ## To prevent nans ## TODO: Review.
    
    # Shape of pdf --> (N_rays, N_coarse). TODO: Review keepdims
    pdf = weights / tf.sum(weights * bin_widths, axis = 1, keepdims = True)
    N_rays = pdf.shape[0]

    # Shape of agg --> (N_rays, N_coarse)
    agg = tf.cumsum(pdf, axis = 1)
    
    # Shape of agg --> (N_rays, N_coarse + 1)
    agg = tf.concat(
        [tf.zeros((agg.shape[0], 1), dtype = agg.dtype), agg], 
        axis = -1
    )

    ## TODO: Use det from params and give it a different name!
    if det:
        spaced = tf.linspace(0, 1, params.sampling.N_fine)
        u_vals = tf.broadcast_to(spaced, (N_rays, params.sampling.N_fine))
    else:
        u_vals = tf.random.uniform(shape = (N_rays, params.sampling.N_fine))

    # Shape of u_vals --> (N_rays, N_fine)
    # Shape of piece_idxs --> (N_rays, N_fine)
    piece_idxs = tf.searchsorted(agg, u_vals, side = 'right')

    #################################################################
    ## TODO: Choose tf.gather or tf.gather_nd

    #################################################################
    ## Using tf.gather
    #################################################################

    ## TODO: I think this code would work but need to check! 
    ## Do not use without testing!
    
    ## agg_, pdf_ and left_edges_ have shape (N_rays, N_fine)
    agg_ = tf.gather(agg, piece_idxs, axis = 1, batch_dims = 1)
    pdf_ = tf.gather(pdf, piece_idxs, axis = 1, batch_dims = 1)
    left_edges_ = tf.gather(left_edges, piece_idxs, axis = 1, batch_dims = 1)

    #################################################################
    ## Using tf.gather_nd
    #################################################################

    ## TODO: If using tf.gather_nd, elaborate shapes.

    # row_idxs = tf.reshape(tf.range(0, N_rays), (N_rays, 1))
    # row_idxs = tf.broadcast_to(row_idxs, (N_rays, params.sampling.N_fine))
    # idxs = tf.stack([row_idxs, piece_idxs], axis = -1)

    # agg_ = tf.gather_nd(agg, idxs)
    # pdf_ = tf.gather_nd(pdf, idxs)
    # left_edges_ = tf.gather_nd(left_edges, idxs)

    #################################################################

    ## Instead of setting denom to 1, trying new logic!
    mask = tf.where(pdf_ < 1e-8, tf.zeros_like(pdf_), tf.ones_like(pdf_))
    pdf_ = tf.maximum(pdf_, 1e-8)

    # Getting samples. TODO: Elaborate.
    # Shape of t_vals_fine --> (N_rays, N_coarse)
    t_vals_fine = (((u_vals - agg_) / pdf_) * mask) + left_edges_
    
    # TODO: Maybe analyze below line?
    t_vals_fine = tf.stop_gradient(t_vals_fine)

    # Shape of t_vals --> (N_rays, N_coarse + N_fine) 
    t_vals = tf.concat([t_vals_coarse, t_vals_fine], axis = 1)
    
    ## TODO: Why sorting? Is it for alpha compositing or something?
    t_vals = tf.sort(t_vals, axis = 1)

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz --> (N_rays, N_coarse + N_fine, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    
    # Shape of rays_d_broadcasted --> (N_rays, N_coarse + N_fine, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d, xyz.shape)
    
    # Shape of dir_inputs --> (N_rays * (N_coarse + N_fine), 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    
    # Shape of xyz_inputs --> (N_rays * (N_coarse + N_fine), 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))

    return xyz_inputs, dir_inputs

if __name__ == '__main__':

    rays_o, rays_d = get_rays(H = 500, W = 500, focal = 250, c2w = np.eye(4))
