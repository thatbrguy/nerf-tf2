import cv2
import numpy as np
import tensorflow as tf

def make_4x4(RT):
    """
    Converts a 3x4 RT matrix into a 4x4 RT matrix. The last 
    row of the 4x4 matrix will be [0, 0, 0, 1].

    TODO: Args, Returns?
    """
    assert RT.shape == (3, 4)

    # The last row of np.eye(4) is [0, 0, 0, 1]. Hence, we 
    # create np.eye(4) and overwrite the first three rows of 
    # it with the RT matrix to get our desired output.
    output = np.eye(4)
    output[:3, :] = RT

    return output

def make_homogeneous(points):
    """
    Converts a points array of shape (N, 3) to (N, 4). The 
    last column will have ones.

    TODO: Args, Returns?
    """
    assert points.shape[1] == 3

    ones = np.ones((points.shape[0], 1), dtype = np.float64)
    output = np.concatenate([points, ones], axis = 1)

    return output

def normalize(vec):
    """
    Normalizes vectors or a batch of vectors.

    If the input (vec) has shape (D,), it is assumed to 
    be D-Dimensional vector. In this case, the output 
    will also have shape (D,). The output will be a unit vector.

    If the input (vec) has shape (N, D), it is assumed to 
    be a batch of N vectors each of which has D-Dimensions. In 
    this case, the output will also have shape (N, D). Each of 
    the N vectors in the output will be a unit vector.

    Args:
        vec         :   A NumPy array of shape (D,) or (N, D)

    Returns
        norm_vec    :   A NumPy array with the same shape as 
                        of the input.

    ## TODO: Consider adding EPS only if magnitude < EPS ?
    """
    assert len(vec.shape) <= 2
    EPS = 1e-8
    
    if len(vec.shape) == 1:
        magnitude = np.sqrt(np.sum(vec ** 2))
        norm_vec = vec / (magnitude + EPS)

    elif len(vec.shape) == 2:
        ## TODO: Verify calculation.
        magnitude = np.sqrt(np.sum(vec ** 2, axis = 1))
        norm_vec = vec / (magnitude[:, None] + EPS)

    return norm_vec

def rotate_vectors(mat, vectors):
    """
    TODO: Elaborate.

    If trasform is (3, 4) or (4, 4), then mat[:3, :3] must 
    contain the rotation matrix. TODO: Cleanup.
    """
    check_1 = mat.shape == (3, 3)
    check_2 = mat.shape == (3, 4)
    check_3 = mat.shape == (4, 4)

    assert np.any([check_1, check_2, check_3]), (
        "Shape of mat is invalid. Must be "
        "either (3, 3), (3, 4) or (4, 4)"
    )
    assert vectors.shape[1] == 3

    transform = mat[:3, :3]
    output = (transform @ vectors.T).T

    return output

def transform_points(RT, points):
    """
    Applies the transformation to the given points.

    TODO: Elaborate.
    """
    assert (RT.shape == (4, 4)) or (RT.shape == (3, 4)), (
        "Shape of the RT matrix is invalid. Must be "
        "either (3, 4) or (4, 4)"
    )
    assert points.shape[1] == 3

    if RT.shape == (3, 4):
        transform = make_4x4(RT)
    elif RT.shape == (4, 4):
        transform = RT

    homogeneous_pts = make_homogeneous(points)
    output = (transform @ homogeneous_pts.T).T

    # Removing the last column which will be all ones.
    output = output[:, :3]

    return output

def batched_transform_points(RT_matrices, points):
    """
    Applies all the RT_matrices on the given points.

    TODO: Support (M, 3, 4) for RT_matrices also.
    """
    assert RT_matrices.shape[1:] == (4, 4), (
        "Shape of the RT matrix is invalid. Must be (M, 4, 4)"
    )
    assert points.shape[1] == 3

    homogeneous_pts = make_homogeneous(points)
    output = (RT_matrices @ homogeneous_pts.T).transpose(0, 2, 1)

    # Removing the last column which will be all ones.
    output = output[..., :3]

    return output

def transform_line_segments(RT, lines):
    """
    Transforms line segments.
    
    Args:
        RT      :   TODO, Explain.
        lines   :   NumPy array of shape (N, 2, 3).

    TODO: Elaborate
    """
    assert (lines.shape[1] == 2) and (lines.shape[2] == 3)
    N = lines.shape[0]
    points = lines.reshape(-1, 3)

    out_points = transform_points(RT, points)
    output = out_points.reshape(N, 2, 3)

    return output

def batched_transform_line_segments(RT_matrices, lines):
    """
    Transforms line segments.
    
    Args:
        RT_matrices :   TODO, Explain.
        lines       :   NumPy array of shape (N, 2, 3).

    TODO: Elaborate
    """
    assert (lines.shape[1] == 2) and (lines.shape[2] == 3)
    M, N = RT_matrices.shape[0], lines.shape[0]
    points = lines.reshape(-1, 3)

    out_points = batched_transform_points(RT_matrices, points)
    output = out_points.reshape(M, N, 2, 3)

    return output

def calculate_new_world_pose(poses, origin_method, basis_method):
    """
    Given N pose matrices (poses), each of which can transform a 
    point from a camera coordinate system to an arbitrary world 
    coordinate system (W1), this function configures a new world 
    coordinate system (W2) based on the given pose matrices (poses).

    TODO: Rewrite if needed.

    Args:
        poses                   :   A NumPy array of shape (N, 4, 4)
        origin_method           :   A string which is either "average", 
                                    "min_dist_solve" or "min_dist_opt"
        basis_method            :   A string which is either "identity" 
                                    or "compute"

    Returns:
        W2_to_W1_transform       :   A NumPy array of shape (4, 4)
    """
    # Shape of origin: (3,)
    origin = compute_new_world_origin(poses, method = origin_method)
    
    # Shape of x_basis, y_basis and z_basis: (3,)
    if basis_method == "identity":
        x_basis = np.array([1.0, 0.0, 0.0])
        y_basis = np.array([0.0, 1.0, 0.0])
        z_basis = np.array([0.0, 0.0, 1.0])

    elif basis_method == "compute":
        x_basis, y_basis, z_basis = compute_new_world_basis(poses)

    else:
        raise ValueError(f"Invalid basis_method: {basis_method}")

    W2_to_W1_3x4 = np.stack([x_basis, y_basis, z_basis, origin], axis = 1)
    W2_to_W1_transform = make_4x4(W2_to_W1_3x4)
    W1_to_W2_transform = np.linalg.inv(W2_to_W1_transform)
    
    return W2_to_W1_transform

def calculate_scene_scale(
        poses, bounds, bounds_method, 
        intrinsics = None, height = None, width = None
    ):
    """
    TODO: Docstring
    """
    rays_o = poses[:, :3, 3]
    rays_d = poses[:, :3, 2]
    far = bounds[:, 1]

    points_far = rays_o + far[:, None] * rays_d

    if bounds_method == "include_corners":
        check_1 = intrinsics is not None
        check_2 = height is not None
        check_3 = width is not None
        assert check_1 and check_2 and check_3
        
        corner_points = get_corner_ray_points(
            poses = poses, bounds = bounds, 
            intrinsics = intrinsics, height = height, width = width
        )

        # We want XYZ coordinates of points_far, rays_o and corner_points 
        # to be within the range [-1, 1]
        points = np.concatenate([rays_o, points_far, corner_points], axis = 0)

    elif bounds_method == "central_ray":
        
        # We want XYZ coordinates of points_far and the origin to be 
        # within the range [-1, 1]
        points = np.concatenate([rays_o, points_far], axis = 0)

    else:
        raise ValueError(f"Invalid bounds_method: {bounds_method}")

    ## TODO: Explain!
    projs = np.abs(points).max(axis = 0)
    largest_proj = projs.max()
    scene_scale_factor = 1 / largest_proj

    return scene_scale_factor

def reconfigure_poses(old_poses, W2_to_W1_transform):
    """
    TODO: Docstring
    """
    # The matrix old_poses[i] would take a point from the i-th camera 
    # coordinate system to the old world (W1) coordinate system. The 
    # matrix W2_to_W1_transform would take a point from the old world 
    # coordinate system (W1) to the new world coordinate system (W2). 
    # Hence, the matrix new_poses[i] would take a point from the i-th 
    # camera coordinate system to the new world coordinate system (W2). 
    new_poses = W2_to_W1_transform @ old_poses

    return new_poses

def reconfigure_scene_scale(old_poses, old_bounds, scene_scale_factor):
    """
    TODO: Elaborate
    """
    if scene_scale_factor >= 1:
        # No scaling required for this case.
        new_poses = old_poses
        new_bounds = old_bounds

    elif scene_scale_factor < 1:
        # Poses and bounds are scaled in this case.
        ## TODO: Verify logic!
        scale_transform = np.eye(4) * scene_scale_factor
        scale_transform[3, 3] = 1

        new_poses = scale_transform @ old_poses.copy()
        new_bounds = old_bounds.copy() * scene_scale_factor

    return new_poses, new_bounds

def scale_imgs_and_intrinsics(old_imgs, old_intrinsics, scale_factor):
    """
    Scales images and intrinsics by the given scaling factor.

    TODO: Elaborate.
    """
    if scale_factor is None:
        new_imgs = old_imgs
        new_intrinsics = old_intrinsics

    elif scale_factor is not None:
        sx, sy = scale_factor, scale_factor
        new_imgs, new_intrinsics = [], []

        ## TODO: Maybe add arg to choose interpolation?
        for idx in range(len(old_imgs)):

            img = cv2.cvtColor(old_imgs[idx].copy(), cv2.COLOR_RGB2BGR)
            resized = cv2.resize(
                img, dsize = None, fx = sx, fy = sy, 
                interpolation = cv2.INTER_AREA,
            )
            new_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            temp = old_intrinsics[idx].copy()
            temp[0, 0], temp[0, 2] = temp[0, 0] * sx, temp[0, 2] * sx
            temp[1, 1], temp[1, 2] = temp[1, 1] * sy, temp[1, 2] * sy
            new_intrinsic = temp

            ## TODO: Maybe add validate intrinsic here for sanity?
            new_imgs.append(new_img)
            new_intrinsics.append(new_intrinsic)

        new_intrinsics = np.array(new_intrinsics)

    else:
        raise ValueError("Invalid setting of scale_factor.")

    return new_imgs, new_intrinsics

def get_corner_ray_points(poses, bounds, intrinsics, height, width):
    """
    TODO: Elaborate

    NOTE: poses must be new_poses

    intrinsics --> (N, 3, 3)
    poses --> (N, 4, 4)
    bounds --> (N, 2)
    """
    H, W = height, width

    # Shape of rays_0: (N, 3)
    rays_o = poses[:, :3, 3]
    # Shape of far: (N, 1)
    far = bounds[:, 1:2]
    
    # Shape of u and v: (1, 4)
    u = np.array([[0, 0, W, W]], dtype = np.float64)
    v = np.array([[0, H, 0, H]], dtype = np.float64)

    # Shape of fu, fv, cu, cv: (N, 1)
    fu, fv = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]
    cu, cv = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]

    # Shape of x_vals, y_vals, z_vals: (N, 4)
    x_vals = (u - cu) / fu
    y_vals = (v - cv) / fv
    z_vals = np.ones(x_vals.shape, dtype = np.float64)
    
    # Shape of directions: (N, 4, 3)
    dirs = np.stack([x_vals, y_vals, z_vals], axis = -1)
    
    output = []
    for idx in range(len(poses)):
        # (4, 4)
        c2w = poses[idx]
        # (4, 3)
        current_dirs = normalize(dirs[idx])
        # (4, 3)
        rotated_dirs = rotate_vectors(c2w, current_dirs)
        # (4, 3)
        normalized_dirs = normalize(rotated_dirs)
        # (1, 3)
        o_vec = rays_o[idx:(idx+1)]
        # (4, 3)
        corner_points = o_vec + far[idx] * normalized_dirs
        output.append(corner_points)

    output = np.array(output)

    # (N*4, 3)
    output = output.reshape(-1, 3)

    return output

def optimize_min_dist_point(poses):
    """
    Optimizes a 3D point such that the perpendicular distance from that 
    point to each of N lines is as small as as possible.
    
    TODO: 
        1.  Elaborate explanation.
        2.  Move some hyperparams to the param file if this function 
            will be used in the future.
    """
    # Initializing the point with all zeros.
    point = tf.Variable([[0., 0., 0.]], dtype = tf.float64)
    
    # Getting the origin point of each line and the normalized 
    # direction vector of each line.
    origins = poses[:, :3, 3]
    directions = normalize(poses[:, :3, 2])

    # Converting origins and directions to TF constants. 
    # (TODO: Should I call them eager tensors?)
    tf_origins = tf.constant(origins)
    tf_directions = tf.constant(directions)
    
    opt = tf.keras.optimizers.SGD(learning_rate = 0.0001)
    losses = []
    
    ## TODO: Verify logic and add comments if needed.
    for _ in range(1000):
        with tf.GradientTape() as tape:
            factor_1 = point - tf_origins
            factor_2 = tf.reduce_sum(factor_1 * factor_1, axis = 1)
            factor_3 = tf.reduce_sum(factor_1 * tf_directions, axis = 1) ** 2
            
            ## loss is the sum of the squared distance to 
            ## each line.
            loss = tf.reduce_sum(factor_2 - factor_3)

        gradient = tape.gradient(loss, point)
        opt.apply_gradients(zip([gradient], [point]))
        losses.append(loss.numpy())

    output = point.numpy().T
    output = np.squeeze(output)

    return output

def solve_min_dist_point(poses):
    """
    This function uses linear algebra to solve for a point in 3D space 
    that has the smallest perpendicular distance to each of N lines.
    
    TODO: Elaborate.
    """
    origins = poses[:, :3, 3]
    directions = normalize(poses[:, :3, 2])

    A_matrix = np.zeros((3, 3))
    b_matrix = np.zeros((3, 1))
    
    ## TODO: Verify logic and add comments if needed.
    for i in range(len(poses)):
        outer = directions[i, :, None] @ directions[i, :, None].T
        factor = outer - np.eye(3)
        A_matrix += factor
        b_matrix += (factor @ origins[i, :, None])

    output = np.linalg.solve(A_matrix, b_matrix)
    output = np.squeeze(output)
    
    return output

def compute_new_world_origin(poses, method):
    """
    Computes the origin of a new world coordinate system given 
    N pose matrices.

    Each of the N pose matrices can transform a point from a camera 
    coordinate system to a arbitrary world coordinate system (W1). 
    In this function, we want to compute the origin of a new 
    world coordinate system (W2).
    
    Args:
        poses       :   A NumPy array of shape (N, 4, 4)
        method      :   A string which is either "average", 
                        "min_dist_solve" or "min_dist_opt"

    Returns:
        origin      :   A NumPy array of shape (3,) which denotes 
                        the location of the new world world coordinate 
                        system (W2).
    """
    if method == "average":
        cam_locs = poses[:, :3, 3]
        origin = np.mean(cam_locs, axis = 0)

    elif method == "min_dist_solve":
        origin = solve_min_dist_point(poses)

    elif method == "min_dist_opt":
        origin = optimize_min_dist_point(poses)
    
    else:
        raise ValueError(f"Invalid method: {method}")

    return origin

def compute_new_world_basis(poses):
    """
    Computed the basis vectors of the new world coordinate system (W2). 
    The basis vectors are represented in the current world coordinate 
    system (W1).

    Args:
        poses       :   A NumPy array of shape (N, 4, 4)

    Returns:
        x_basis     :   A NumPy array of shape (3,) which is a vector in 
                        the current world system (W1) that represents the 
                        X-Direction of the new world coordinate system (W2).

        y_basis     :   A NumPy array of shape (3,) which is a vector in 
                        the current world system (W1) that represents the 
                        Y-Direction of the new world coordinate system (W2).

        z_basis     :   A NumPy array of shape (3,) which is a vector in 
                        the current world system (W1) that represents the 
                        Z-Direction of the new world coordinate system (W2).
    """
    
    # Computing the average Y-Axis direction of the cameras.
    avg_y_direction = np.mean(poses[:, :3, 1], axis = 0)
    # Normalizing the vector to get a unit vector.
    avg_y_direction = normalize(avg_y_direction)

    # Setting the y_basis of the new new world coordinate 
    # system (W2) as avg_y_direction.
    y_basis = avg_y_direction

    # We would now like to calculate the z_basis vector. The z_basis 
    # vector must be perpendicular to the y_basis vector. We can ensure 
    # this by obtaining z_basis as a result of performing cross product 
    # of some "reference" vector with the y_basis vector. Here, we take 
    # the X-Axis direction of the first camera in the poses array to serve 
    # as this "reference" vector. You can choose this "reference" vector 
    # in many other ways as well; this method is just my preference. 
    # In the below code, temp_vector is the "reference" vector.
    temp_vector = poses[0, :3, 0]
    # Normalizing the vector to get a unit vector.
    temp_vector = normalize(temp_vector)

    # Using temp_vector and y_basis to get z_basis. Note that 
    # z_basis is perpendicular to temp_vector and y_basis.
    z_basis = np.cross(temp_vector, y_basis)
    # Normalizing the vector to get a unit vector.
    z_basis = normalize(z_basis)

    # Using y_basis and z_basis to get x_basis. Note that 
    # x_basis, y_basis and z_basis are now mutually perpendicular.
    x_basis = np.cross(y_basis, z_basis)
    # Normalizing the vector to get a unit vector.
    x_basis = normalize(x_basis)

    return x_basis, y_basis, z_basis
