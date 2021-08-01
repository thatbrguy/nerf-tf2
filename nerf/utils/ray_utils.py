import tensorflow as tf
import numpy as np

from nerf.utils import pose_utils

def get_rays(H, W, intrinsic, c2w):
    """
    Gets ray origin and ray directions in the world coordinate system.

    TODO: Elaborate.

    TODO: Think about offset by 0.5 so that rays go through 
    the middle of the pixel.
    """
    H_vals = np.arange(H, dtype = np.float64)
    W_vals = np.arange(W, dtype = np.float64)
    u, v = np.meshgrid(W_vals, H_vals, indexing = "xy")

    ## TODO: Explain logic.
    x_vals = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y_vals = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    z_vals = np.ones(x_vals.shape, dtype = np.float64)

    directions = np.stack([x_vals, y_vals, z_vals], axis = -1)
    # (H, W, 3) --> (H*W, 3) TODO: Verify
    directions = directions.reshape(-1, 3)

    ## TODO: Check output for correctness! Add comments!
    rays_d = pose_utils.rotate_vectors(c2w, directions)
    rays_d = pose_utils.normalize(rays_d)
    
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    
    return rays_o, rays_d

def get_rays_tf(H, W, intrinsic, c2w):
    """
    Gets ray origin and ray directions in the world coordinate system.

    TODO: Elaborate.

    TODO: Think about offset by 0.5 so that rays go through 
    the middle of the pixel.
    """
    EPS = 1e-8
    H_vals = tf.range(start = 0, limit = H, dtype = tf.float32)
    W_vals = tf.range(start = 0, limit = W, dtype = tf.float32)
    u, v = tf.meshgrid(W_vals, H_vals, indexing = "xy")

    ## TODO: Explain logic.
    x_vals = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y_vals = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    z_vals = tf.ones_like(x_vals)

    directions = tf.stack([x_vals, y_vals, z_vals], axis = -1)
    # (H, W, 3) --> (H*W, 3) TODO: Verify
    directions = tf.reshape(directions, (-1, 3))

    ## TODO: Check output for correctness! Add comments!
    rotation = c2w[:3, :3]

    # (A @ B.T).T == (B @ A.T)
    rays_d = tf.linalg.matmul(directions, tf.transpose(rotation))

    # Normalizing rays_d
    ## TODO: Check if keepdims performs as expected.
    magnitude = tf.sqrt(tf.reduce_sum(rays_d ** 2, axis = 1, keepdims = True))
    rays_d = rays_d / (magnitude + EPS)
    
    rays_o = tf.broadcast_to(c2w[:3, 3], tf.shape(rays_d))
    
    return rays_o, rays_d

def create_input_batch_coarse_model(params, rays_o, rays_d, near, far):
    """
    Given rays_o and rays_d, creates a batch of inputs that 
    can be fed to the coarse model.

    Args:
        rays_o      : TODO (type, explain) with shape (N_rays, 3).
        rays_d      : TODO (type, explain) with shape (N_rays, 3).
        near        : TODO (type, explain) with shape (N_rays, 1).
        far         : TODO (type, explain) with shape (N_rays, 1).

    Returns:
        xyz_inputs  : TODO: Explain
        dir_inputs  : TODO: Explain

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
        u_vals = tf.random.uniform(shape = tf.shape(bin_widths))

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
    rays_d_broadcasted = tf.broadcast_to(rays_d[:, None, :], tf.shape(xyz))
    
    # Shape of dir_inputs --> (N_rays * N_coarse, 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    # Shape of xyz_inputs --> (N_rays * N_coarse, 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))

    # Collecting bin_data for returning.
    bin_data = {
        "bin_edges": bin_edges, "left_edges": left_edges,
        "right_edges": right_edges, "bin_widths": bin_widths,
    }

    data = {
        "xyz_inputs": xyz_inputs,
        "dir_inputs": dir_inputs,
        "bin_data": bin_data,
        "t_vals": t_vals,
    }
    
    return data

def create_input_batch_fine_model(params, rays_o, rays_d, bin_weights, bin_data, t_vals_coarse):
    """
    Creates batch of inputs for the fine model.

    TODO: Elaborate!

    TODO: Consider explaining concepts briefly over certain 
    lines. Detailed description can be provided elsewhere.
    """
    # Extracting useful content from bin_data
    # Shape of left_edges and bin_widths --> (N_rays, N_coarse)
    left_edges = bin_data["left_edges"]
    bin_widths = bin_data["bin_widths"]

    # Creating pdf from weights.
    # Shape of bin_weights --> (N_rays, N_coarse)
    bin_weights = bin_weights + 1e-5 ## To prevent nans ## TODO: Review.
    
    # Shape of pdf --> (N_rays, N_coarse).
    denom = tf.reduce_sum(bin_weights * bin_widths, axis = 1, keepdims = True)
    pdf = bin_weights / denom

    # Shape of agg --> (N_rays, N_coarse)
    agg = tf.cumsum(pdf * bin_widths, axis = 1)
    
    # Shape of agg (output) --> (N_rays, N_coarse + 1)
    agg = tf.concat([tf.zeros_like(agg[:, 0:1]), agg], axis = -1)

    ## TODO: Use det from params and give it a different name!
    det = False
    N_rays = tf.shape(pdf)[0]
    
    if det:
        spaced = tf.linspace(0, 1, params.sampling.N_fine)
        u_vals = tf.broadcast_to(spaced, (N_rays, params.sampling.N_fine))
    else:
        u_vals = tf.random.uniform(shape = (N_rays, params.sampling.N_fine))

    ## TODO: Verify functionality.
    # Shape of u_vals --> (N_rays, N_fine)
    # Shape of piece_idxs --> (N_rays, N_fine)
    # Shape of agg[:, 1:-1] --> (N_rays, N_fine - 1).
    
    # agg[:, 1:-1] has the "inner edges" with which we can do 
    # searchsorted correctly. 
    piece_idxs = tf.searchsorted(agg[:, 1:-1], u_vals, side = 'right')

    ## agg_, pdf_ and left_edges_ have shape (N_rays, N_fine)
    agg_ = tf.gather(agg[:, :-1], piece_idxs, axis = 1, batch_dims = 1)
    pdf_ = tf.gather(pdf, piece_idxs, axis = 1, batch_dims = 1)
    left_edges_ = tf.gather(left_edges, piece_idxs, axis = 1, batch_dims = 1)

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
    
    ## TODO: Mention reason for sorting.
    t_vals = tf.sort(t_vals, axis = 1)

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz --> (N_rays, N_coarse + N_fine, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    
    # Shape of rays_d_broadcasted --> (N_rays, N_coarse + N_fine, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d[:, None, :], tf.shape(xyz))
    
    # Shape of dir_inputs --> (N_rays * (N_coarse + N_fine), 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    
    # Shape of xyz_inputs --> (N_rays * (N_coarse + N_fine), 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))

    data = {
        "xyz_inputs": xyz_inputs,
        "dir_inputs": dir_inputs,
        "t_vals": t_vals,
    }

    return data

def sigma_to_alpha(sigma, diffs):
    """
    Computes alpha.
    TODO: Elaborate.
    
    Args:
        sigma   : TODO (type, explain) with shape (N_rays, N_samples)
        diffs   : TODO (type, explain) with shape (N_rays, N_samples)

    Returns:
        alpha   : TODO (type, explain) with shape (N_rays, N_samples)

    """
    ## TODO: Add noise to sigma based on parameter setting?
    alpha = 1 - tf.exp(-sigma * diffs)
    return alpha

def compute_weights(sigma, t_vals, N_samples):
    """
    Computes weights. A weight value w_i is the product of T_i 
    and alpha_i. TODO: Elaborate.
    
    This function can be used for both the coarse and fine 
    models. If used for the coarse model, these weight values also serve as 
    the weight of each bin. TODO: Elaborate and refactor this.

    Args:
        sigma       : TODO (type, explain) with shape (N_rays * N_samples, 1)
        t_vals      : TODO (type, explain) with shape (N_rays, N_samples)
        N_samples   : Integer. TODO: Explain.

    Returns:
        weights     : TODO (type, explain) with shape (N_rays, N_samples)
    """
    EPS = 1e-10
    INF = 1e10
    
    # Shape of diffs --> (N_rays, N_samples - 1)
    diffs = t_vals[:, 1:] - t_vals[:, :-1]

    ## TODO: Should it be INF or can we just provide the far bound value?
    N_rays = tf.shape(diffs)[0]
    last_val_array = tf.fill((N_rays, 1), INF)

    # Shape of diffs_ --> (N_rays, N_samples)
    # NOTE: We do not need to multiply diffs_ with the magnitude 
    # of the ray_d vectors. TODO: Elaborate.
    diffs_ = tf.concat([diffs, last_val_array], axis = -1)

    # Shape of sigma_ --> (N_rays, N_samples)
    sigma_ = tf.reshape(tf.squeeze(sigma), (-1, N_samples))
    # Shape of alpha --> (N_rays, N_samples)
    alpha = sigma_to_alpha(sigma_, diffs_)

    # TODO: Provide explanation to the following somewhere if possible.
    # Shape of weights --> (N_rays, N_samples).
    weights = alpha * tf.math.cumprod(1 - alpha + EPS, axis = 1, exclusive = True)
    
    return weights

def post_process_model_output(sample_rgb, sigma, t_vals, white_bg = False):
    """
    TODO: Docstring.
    
    N_samples --> (N_coarse) or (N_coarse + N_fine)

    Args:
        sample_rgb      : TODO (type, explain) with shape (N_rays * N_samples, 3)
        sigma           : TODO (type, explain) with shape (N_rays * N_samples, 1)
        t_vals          : TODO (type, explain) with shape (N_rays, N_samples)

    Returns:
        TODO

    TODO: Check consistent nomenclature usage across the codebase (Ex: when to 
    use "bin" and when to use "sample".)
    """
    post_proc_model_outputs = dict()
    N_samples = tf.shape(t_vals)[1]

    # Shape of weights --> (N_rays, N_samples).
    weights = compute_weights(sigma, t_vals, N_samples)

    # Shape of sample_rgb_ --> (N_rays, N_samples, 3).
    sample_rgb_ = tf.reshape(sample_rgb, (-1, N_samples, 3))

    # Shape of pred_rgb --> (N_rays, 3)
    pred_rgb = tf.reduce_sum(weights[..., None] * sample_rgb_, axis = 1)

    # Shape of pred_depth --> (N_rays,); TODO: Verify shape.
    pred_depth = tf.reduce_sum(t_vals * weights, axis = 1)

    # Shape of acc_map --> (N_rays,); TODO: Verify shape.
    acc_map = tf.reduce_sum(weights, axis = 1)

    if white_bg:
        # RGB colors are between [0, 1] in pred_rgb. I think the 
        # below equation assumes that pred_rgb is "pre-multiplied" 
        # (i.e. multiplied with a "mask" already). White background 
        # has color value of 1 in all three channels and so
        # (1 - acc_map[:, None]) * 1 = (1 - acc_map[:, None]).
        ## TODO: Verify, clarify, elaborate, move explanation 
        ## to somewhere else maybe etc.
        pred_rgb = pred_rgb + (1 - acc_map[:, None]) 

    post_proc_model_outputs["weights"] = weights
    post_proc_model_outputs["pred_rgb"] = pred_rgb
    post_proc_model_outputs["pred_depth"] = pred_depth
    
    return post_proc_model_outputs

if __name__ == '__main__':

    rays_o, rays_d = get_rays(H = 500, W = 500, focal = 250, c2w = np.eye(4))
