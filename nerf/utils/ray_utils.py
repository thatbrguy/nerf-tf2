import tensorflow as tf
import numpy as np

from nerf.utils import pose_utils

def get_rays(H, W, intrinsic, c2w):
    """
    Returns the origin and direction vectors of the desired rays.
    
    We have one ray passing through each pixel of an image with 
    height "H" and width "W". Hence, in total we have H*W rays.
    The aforementioned image was captured with a camera which 
    has the pose matrix "c2w" and intrinsic matrix "intrinsic".
    
    Args:
        H           :   An integer representing the height of the image.
        W           :   An integer representing the width of the image.
        intrinsic   :   A NumPy array of shape (3, 3) representing 
                        the intrinsic matrix.
        c2w         :   A NumPy array of shape (4, 4) representing 
                        the pose of the camera.

    Returns:
        rays_o      :   A NumPy array of shape (H * W, 3) representing 
                        the origin vector of each ray.
        rays_d      :   A NumPy array of shape (H * W, 3) representing 
                        the normalized direction vector of each ray.
    """
    H_vals = np.arange(H, dtype = np.float64)
    W_vals = np.arange(W, dtype = np.float64)
    u, v = np.meshgrid(W_vals, H_vals, indexing = "xy")

    # Getting points in the camera space from points in the image space. 
    # z_vals is set to 1 for all points since we are anyway only interested 
    # in obtaining direction vectors, and the direction vectors will be 
    # normalized in a future step.
    x_vals = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y_vals = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    z_vals = np.ones(x_vals.shape, dtype = np.float64)
    
    directions = np.stack([x_vals, y_vals, z_vals], axis = -1)    
    directions = directions.reshape(-1, 3)

    # Rotating and normalizing the direction vectors
    rays_d = pose_utils.rotate_vectors(c2w, directions)
    rays_d = pose_utils.normalize(rays_d)
    
    # The origin vector is the same for all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    
    return rays_o, rays_d

def get_rays_tf(H, W, intrinsic, c2w):
    """
    Returns the origin and direction vectors of the desired rays.
    
    We have one ray passing through each pixel of an image with 
    height "H" and width "W". Hence, in total we have H*W rays.
    The aforementioned image was captured with a camera which 
    has the pose matrix "c2w" and intrinsic matrix "intrinsic".

    This function uses TensorFlow. 
    
    Args:
        H           :   A TensorFlow tensor representing the height of the image.
        W           :   A TensorFlow tensor representing the width of the image.
        intrinsic   :   A TensorFlow tensor of shape (3, 3) representing 
                        the intrinsic matrix.
        c2w         :   A TensorFlow tensor of shape (4, 4) representing 
                        the pose of the camera.

    Returns:
        rays_o      :   A TensorFlow tensor of shape (H*W, 3) representing 
                        the origin vector of each ray.
        rays_d      :   A TensorFlow tensor of shape (H*W, 3) representing 
                        the normalized direction vector of each ray.
    """
    EPS = 1e-8
    H_vals = tf.range(start = 0, limit = H, dtype = tf.float32)
    W_vals = tf.range(start = 0, limit = W, dtype = tf.float32)
    u, v = tf.meshgrid(W_vals, H_vals, indexing = "xy")

    # Getting points in the camera space from points in the image space. 
    # z_vals is set to 1 for all points since we are anyway only interested 
    # in obtaining direction vectors, and the direction vectors will be 
    # normalized in a future step.
    x_vals = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y_vals = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    z_vals = tf.ones_like(x_vals)

    directions = tf.stack([x_vals, y_vals, z_vals], axis = -1)
    directions = tf.reshape(directions, (-1, 3))

    # Rotating the direction vectors.
    # Note that: transpose(A @ transpose(B)) == (B @ transpose(A))
    rotation = c2w[:3, :3]
    rays_d = tf.linalg.matmul(directions, tf.transpose(rotation))

    # Normalizing the direction vectors.
    magnitude = tf.sqrt(tf.reduce_sum(rays_d ** 2, axis = 1, keepdims = True))
    rays_d = rays_d / (magnitude + EPS)

    # The origin vector is the same for all rays.
    rays_o = tf.broadcast_to(c2w[:3, 3], tf.shape(rays_d))
    
    return rays_o, rays_d

def create_depth_map(
    pred_depth, H, W, scale_factor, map_type, 
    intrinsic = None, C_to_W2 = None
):
    """
    Creates a depth map

    TODO: Finish docstring!
    """
    depth_scale_corrected = pred_depth * (1 / scale_factor)
    
    if map_type == "type_1":
        depth_map = depth_scale_corrected.reshape(H, W)

    elif map_type == "type_2":
        rays_o, rays_d = get_rays(H, W, intrinsic, C_to_W2)
        points_W2 = rays_o + rays_d * depth_scale_corrected[:, None]

        W2_to_C = np.linalg.inv(C_to_W2)
        points_cam = pose_utils.transform_points(W2_to_C, points_W2)

        z_vals = points_cam[:, 2]
        depth_map = z_vals.reshape(H, W)

    else:
        raise ValueError(f"Invalid map_type: {map_type}")

    return depth_map

def create_input_batch_coarse_model(params, rays_o, rays_d, near, far):
    """
    Creates the input data required for the coarse model.

    This function returns a dictionary which contains the input required for 
    the coarse model (xyz_inputs and dir_inputs) as well as some additional 
    information (bin_data and t_vals). The additional information is useful
    for some downstream functionality.

    An important operation performed in this function is the creation of bins
    along each ray. The bins are created for the purpose of stratified sampling.
    N_coarse number of bins are created for each ray. One "t" value is sampled
    in each of the N_coarse bins for each rays. The "t" values are used for the
    creation of the XYZ coordinates that are to be input to the coarse model.

    The information regarding the created bins is stored in the dictionary
    bin_data. The sampled "t" values are stored in t_vals.

    Args:
        params      :   The params object as returned by the function load_params. 
                        The function load_params is defined in params_utils.py
        rays_o      :   A TensorFlow tensor of shape (N_rays, 3) representing the
                        origin vectors of the rays.
        rays_d      :   A TensorFlow tensor of shape (N_rays, 3) representing the
                        normalized direction vectors of the rays.
        near        :   A TensorFlow tensor of shape (N_rays, 1) representing the 
                        near bound for each ray.
        far         :   A TensorFlow tensor of shape (N_rays, 1) representing the 
                        far bound for each ray.

    Returns:
        A dictionary called data containing the following:
        
        xyz_inputs  :   A TensorFlow tensor of shape (N_rays * N_coarse, 3) which can
                        be used as the XYZ coordinates input for the coarse model.
        dir_inputs  :   A TensorFlow tensor of shape (N_rays * N_coarse, 3) which can
                        be used as the direction vectors input for the coarse model.
        bin_data    :   A dictionary containing various items relevant to the bins. 
                        Please refer to the code for more information.
        t_vals      :   A TensorFlow tensor of shape (N_rays, N_coarse). It is 
                        the "t" in the equation r(t) = o + t * d
    """
    # We are interested in creating N_coarse bins between the near bound 
    # and the far bound. To do that, we need N_coarse+1 bin edges (including 
    # the left most bin edge and the right most bin edge).

    if not params.sampling.lin_inv_depth:
        # In this case, the distance between any two adjacent bin edges
        # are the same.

        # Shape of vals: (N_rays, N_coarse + 1, 1)
        vals = tf.linspace(
            start = near, stop = far, 
            num = params.sampling.N_coarse + 1, axis = 1
        )
    else:
        # In this case, the inverse distance between any two adjacent 
        # bin edges are the same. The consequence of this setting will 
        # be explain in the docs soon (TODO: add explanation in docs).

        # TODO: Add eps to avoid avoid division by zero error, or 
        # ensure elsewhere that division by zero error cannot happen.
        # Shape of vals: (N_rays, N_coarse + 1, 1)
        vals = 1 / tf.linspace(
            start = (1 / near), stop = (1 / far), 
            num = params.sampling.N_coarse + 1, axis = 1
        )
    
    # Shape of bin_edges: (N_rays, N_coarse + 1)
    bin_edges = tf.squeeze(vals, axis = 2)

    # left_edges contains the left edge of each of the N_coarse bins, 
    # and right_edges contains the right edge of each of the N_coarse bins.

    # Shape of left_edges: (N_rays, N_coarse)
    left_edges = bin_edges[:, :-1]
    # Shape of right_edges: (N_rays, N_coarse)
    right_edges = bin_edges[:, 1:]

    # Now, we would like to get a sample of "t" from each of the N_coarse bins 
    # for each of the N_rays rays. This can be done with or without perturbation. 

    if params.sampling.perturb:
        # In this case, perturbation is enabled. In this case, for each bin,
        # a "t" value in the bin (including the left bin edge but
        # excluding the right bin edge) is randomly selected
        # according to a uniform distribution.

        # Shape of bin_widths: (N_rays, N_coarse)
        bin_widths = right_edges - left_edges

        # Getting random uniform samples in the range [0, 1). 
        # Shape of u_vals: (N_rays, N_coarse)
        u_vals = tf.random.uniform(shape = tf.shape(bin_widths))

        # Using the below logic, we get one random sample in each bin.
        # Shape of t_vals: (N_rays, N_coarse)
        t_vals = left_edges + (u_vals * bin_widths)

    elif not params.sampling.perturb:
        # In this case, perturbation is NOT enabled. In this case, for each bin, 
        # the middle value of each bin (which is the average of the left bin edge 
        # and right bin edge for each bin) is taken as the "t" value for that bin.

        # Getting the mid point of each bin.
        # Shape of bin_mids: (N_rays, N_coarse)
        bin_mids = 0.5 * (left_edges + right_edges)

        # In this case, we directly use bin_mids as t_vals.
        # Shape of t_vals: (N_rays, N_coarse)
        t_vals = bin_mids

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz: (N_rays, N_coarse, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    # Shape of rays_d_broadcasted: (N_rays, N_coarse, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d[:, None, :], tf.shape(xyz))
    
    # Shape of dir_inputs: (N_rays * N_coarse, 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    # Shape of xyz_inputs: (N_rays * N_coarse, 3)
    xyz_inputs =  tf.reshape(xyz, (-1, 3))

    # Collecting information regarding the bins in bin_data.
    bin_data = {
        "bin_edges": bin_edges, "left_edges": left_edges,
        "right_edges": right_edges, "bin_widths": bin_widths,
    }

    # Creating data for returning.
    data = {
        "xyz_inputs": xyz_inputs,
        "dir_inputs": dir_inputs,
        "bin_data": bin_data,
        "t_vals": t_vals,
    }
    
    return data

def create_input_batch_fine_model(params, rays_o, rays_d, bin_weights, bin_data, t_vals_coarse):
    """
    Creates the input data required for the fine model.

    This function returns a dictionary which contains the input required for 
    the coarse model (xyz_inputs and dir_inputs) as well as some additional 
    information (t_vals). The additional information is useful for some 
    downstream functionality.

    This function implements inverse transform sampling to sample N_fine "t" values
    from a piece-wise constant PDF. The piece-wise constant PDF is created using the
    bin weights. The implementation of the inverse transform sampling is different 
    from the official codebase. A blog post explaining this implementation will be
    created in the future.

    The "t" values that are used for creating the inputs to fine model are a combination
    of the "t" values used for the coarse model AND the "t" values sampled from the 
    piece-wise constant PDF. Hence, a total of (N_coarse + N_fine) "t" values are used
    for creating the input to the fine model.

    Args:
        params          :   The params object as returned by the function load_params. 
                            The function load_params is defined in params_utils.py

        rays_o          :   A TensorFlow tensor of shape (N_rays, 3) representing the
                            origin vectors of the rays.

        rays_d          :   A TensorFlow tensor of shape (N_rays, 3) representing the
                            normalized direction vectors of the rays.

        bin_weights     :   The computed bin weights. This value can be obtained from
                            the output of the function post_process_model_output
                            when post_process_model_output is used to post process
                            the coarse model predictions.

        bin_data        :   A dictionary containing information about the bins along 
                            each ray. This value can be obtained from the output of the 
                            function create_input_batch_coarse_model

        t_vals_coarse   :   A TensorFlow tensor of shape (N_rays, N_coarse). This value 
                            can be obtained from the output of the function 
                            create_input_batch_coarse_model

    Returns:
        A dictionary called data containing the following:
        
        xyz_inputs  :   A TensorFlow tensor of shape (N_rays * (N_coarse + N_fine), 3) 
                        which can be used as the XYZ coordinates input for the fine model.
        dir_inputs  :   A TensorFlow tensor of shape (N_rays * (N_coarse + N_fine), 3) 
                        which can be used as the direction vectors input for the coarse model.
        t_vals      :   A TensorFlow tensor of shape (N_rays, (N_coarse + N_fine)). It is 
                        the "t" in the equation r(t) = o + t * d
    """
    # Shape of left_edges and bin_widths: (N_rays, N_coarse)
    left_edges = bin_data["left_edges"]
    bin_widths = bin_data["bin_widths"]

    # Creating pdf from weights.
    # Shape of bin_weights: (N_rays, N_coarse)
    bin_weights = bin_weights + 1e-5 # 1e-5 is added to prevent potential division by 0 errors.
    
    # Shape of pdf: (N_rays, N_coarse).
    denom = tf.reduce_sum(bin_weights * bin_widths, axis = 1, keepdims = True)
    pdf = bin_weights / denom

    # Shape of agg: (N_rays, N_coarse)
    agg = tf.cumsum(pdf * bin_widths, axis = 1)
    
    # Shape of agg (output): (N_rays, N_coarse + 1)
    agg = tf.concat([tf.zeros_like(agg[:, 0:1]), agg], axis = -1)

    # TODO: Use det from params and give it a different name!
    det = False
    N_rays = tf.shape(pdf)[0]
    
    if det:
        spaced = tf.linspace(0, 1, params.sampling.N_fine)
        u_vals = tf.broadcast_to(spaced, (N_rays, params.sampling.N_fine))
    else:
        u_vals = tf.random.uniform(shape = (N_rays, params.sampling.N_fine))

    ## TODO: Verify functionality.
    # Shape of u_vals: (N_rays, N_fine)
    # Shape of piece_idxs: (N_rays, N_fine)
    # Shape of agg[:, 1:-1]: (N_rays, N_fine - 1).
    
    # agg[:, 1:-1] has the "inner edges" with which we can do 
    # searchsorted correctly. 
    piece_idxs = tf.searchsorted(agg[:, 1:-1], u_vals, side = 'right')

    ## agg_, pdf_ and left_edges_ have shape (N_rays, N_fine)
    agg_ = tf.gather(agg[:, :-1], piece_idxs, axis = 1, batch_dims = 1)
    pdf_ = tf.gather(pdf, piece_idxs, axis = 1, batch_dims = 1)
    left_edges_ = tf.gather(left_edges, piece_idxs, axis = 1, batch_dims = 1)

    # Creating mask so that the edge case that happens when elements of
    # pdf_ is less than 1e-8 can be handled.
    mask = tf.where(pdf_ < 1e-8, tf.zeros_like(pdf_), tf.ones_like(pdf_))
    pdf_ = tf.maximum(pdf_, 1e-8)

    # Shape of t_vals_fine: (N_rays, N_coarse)
    t_vals_fine = (((u_vals - agg_) / pdf_) * mask) + left_edges_
    t_vals_fine = tf.stop_gradient(t_vals_fine)

    # Shape of t_vals: (N_rays, N_coarse + N_fine) 
    t_vals = tf.concat([t_vals_coarse, t_vals_fine], axis = 1)

    # t_vals needs to be sorted since the coarse and fine model 
    # output post processing steps assume that the t_vals are 
    # in ascending order.
    t_vals = tf.sort(t_vals, axis = 1)

    # Getting xyz points using the equation r(t) = o + t * d
    # Shape of xyz: (N_rays, N_coarse + N_fine, 3)
    xyz = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    
    # Shape of rays_d_broadcasted: (N_rays, N_coarse + N_fine, 3)
    rays_d_broadcasted = tf.broadcast_to(rays_d[:, None, :], tf.shape(xyz))
    
    # Shape of dir_inputs: (N_rays * (N_coarse + N_fine), 3)
    dir_inputs = tf.reshape(rays_d_broadcasted, (-1, 3))
    
    # Shape of xyz_inputs: (N_rays * (N_coarse + N_fine), 3)
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

    Note:
        Currently support for adding noise to sigma before computing 
        alpha is not available. TODO: Consider adding support.

    Args:
        sigma   : A TensorFlow tensor with shape (N_rays, N_samples)
        diffs   : A TensorFlow tensor with shape (N_rays, N_samples)

    Returns:
        alpha   : A TensorFlow tensor with shape (N_rays, N_samples)
    """
    alpha = 1 - tf.exp(-sigma * diffs)
    return alpha

def compute_weights(sigma, t_vals, N_samples):
    """
    Computes weights.

    Let:
        1.  weights[k, i] be the weight value of the "t" value i for ray k.

        2.  T[k, i] be the approximated accumulated transmittance (computed
            as per equation 3 of the paper) of the "t" value i for ray k.

        3.  alpha[k, i] be the alpha value (computed using the 
            function sigma_to_alpha) of the "t" value i for ray k.

    Then, weights[i, j] is the product of T[i, j] and alpha[i, j].
    Please refer to equation 3 of the NeRF paper for more information.

    This function can be used for both the coarse and fine models.
    If this function is used for the coarse model, then these weight 
    values can also be used as the weight values of the bins of the rays.

    Args:
        sigma       :   A TensorFlow tensor with shape (N_rays * N_samples, 1)
        t_vals      :   A TensorFlow tensor with shape (N_rays, N_samples)
        N_samples   :   A TensorFlow tensor whose value is either N_coarse 
                        or (N_coarse + N_fine). The value should be N_coarse
                        if this function is used for the coarse model. The 
                        value should be (N_coarse + N_fine) if this function
                        is used for the fine model.

    Returns:
        weights     :   A TensorFlow tensor with shape (N_rays, N_samples)
    """
    EPS = 1e-10
    INF = 1e10

    # Shape of diffs: (N_rays, N_samples - 1)
    diffs = t_vals[:, 1:] - t_vals[:, :-1]

    N_rays = tf.shape(diffs)[0]
    last_val_array = tf.fill((N_rays, 1), INF)

    # Shape of diffs_: (N_rays, N_samples)
    diffs_ = tf.concat([diffs, last_val_array], axis = -1)

    # TODO: Explain why elements in diffs_ is NOT multipled with the norm 
    # of the corresponding vector in rays_d.

    # Shape of sigma_: (N_rays, N_samples)
    sigma_ = tf.reshape(tf.squeeze(sigma), (-1, N_samples))
    # Shape of alpha: (N_rays, N_samples)
    alpha = sigma_to_alpha(sigma_, diffs_)

    # Shape of weights: (N_rays, N_samples).
    # TODO: Provide explanation for the below line of code.
    weights = alpha * tf.math.cumprod(1 - alpha + EPS, axis = 1, exclusive = True)

    return weights

def post_process_model_output(sample_rgb, sigma, t_vals, white_bg = False):
    """
    Computes the expected color and expected depth for each ray.

    Given the predictions of the coarse model or the fine model for
    each "t" value of each ray, this function computes the expected
    color and expected depth for each ray. This function can be used
    for both the coarse model and the fine model.

    We use N_samples to refer to either N_coarse or (N_coarse + N_fine). 
    The value of N_samples should be N_coarse if this function is used for 
    the coarse model. The value should be (N_coarse + N_fine) if this function
    is used for the fine model.

    Do note that pred_depth is the depth as measured in the W3 coordinate system.
    The W3 coordinate system is a scaled version of the W2 coordinate system. 
    To get the depth as measured in the W2 coordinate system, the scale must 
    be accounted for. This will be taken care of in render.py and eval.py 
    (TODO: Add depth prediction for render.py and eval.py)

    TODO: Check consistent nomenclature usage across the codebase (Ex: when to 
    use "bin" and when to use "sample".)

    Args:
        sample_rgb      :   A TensorFlow tensor with shape (N_rays * N_samples, 3)
        sigma           :   A TensorFlow tensor with shape (N_rays * N_samples, 1)
        t_vals          :   A TensorFlow tensor with shape (N_rays, N_samples)
        white_bg        :   A boolean which indicates with a white background 
                            should be assumed. Default is set to False.

    Returns:
        A dictionary called post_proc_model_outputs containing the following:

        weights         :   A TensorFlow tensor with shape (N_rays, N_samples) 
                            representing the weights computed by the function 
                            compute_weights
        pred_rgb        :   A TensorFlow tensor with shape (N_rays, 3) representing 
                            the expected color for each ray.
        pred_depth      :   A TensorFlow tensor with shape (N_rays,) representing
                            the expected depth for each ray.
    """
    post_proc_model_outputs = dict()
    N_samples = tf.shape(t_vals)[1]

    # Shape of weights: (N_rays, N_samples)
    weights = compute_weights(sigma, t_vals, N_samples)

    # Shape of sample_rgb_: (N_rays, N_samples, 3)
    sample_rgb_ = tf.reshape(sample_rgb, (-1, N_samples, 3))

    # Shape of pred_rgb: (N_rays, 3)
    pred_rgb = tf.reduce_sum(weights[..., None] * sample_rgb_, axis = 1)

    # Shape of pred_depth: (N_rays,)
    pred_depth = tf.reduce_sum(t_vals * weights, axis = 1)

    # Shape of acc_map: (N_rays,)
    acc_map = tf.reduce_sum(weights, axis = 1)

    if white_bg:
        # TODO: Provide explanation for the below line of code.
        pred_rgb = pred_rgb + (1 - acc_map[:, None]) 

    post_proc_model_outputs["weights"] = weights
    post_proc_model_outputs["pred_rgb"] = pred_rgb
    post_proc_model_outputs["pred_depth"] = pred_depth
    
    return post_proc_model_outputs

if __name__ == '__main__':
    pass
