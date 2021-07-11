import numpy as np

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

def reconfigure_poses_and_bounds(old_poses, old_bounds, origin_method):
    """
    Given N pose matrices (old_poses), each of which can transform a 
    point from a camera coordinate system to an arbitrary world 
    coordinate system (W1), this function does the following operations:
        
        1. Configures a new world coordinate system (W2) based based on 
        the given pose matrices (old_poses).

        2. Returns N new pose matrices (new_poses), each of which can
        transform a point from a camera coordinate system to the new
        world coordinate system (W2).

    Args:
        old_poses       :   A NumPy array of shape (N, 4, 4)
        old_bounds      :   A NumPy array of shape (N, 2)
        origin_method   :   A string which is either "average" 
                            or "min_dist"

    Returns:
        new_poses       :   A NumPy array of shape (N, 4, 4)

    ## TODO: Consider moving this fuction (and other relavent ones) 
    ## to pose_utils!
    """
    
    # Shape of origin --> (3,)
    origin = compute_new_world_origin(old_poses, method = "average")
    # Shape of x_basis, y_basis and z_basis --> (3,)
    x_basis, y_basis, z_basis = compute_new_world_basis(old_poses)

    W2_to_W1_3x4 = np.stack([x_basis, y_basis, z_basis, origin], axis = 1)
    W2_to_W1_pose = pose_utils.make_4x4(W2_to_W1_3x4)
    W1_to_W2_pose = np.linalg.inv(W2_to_W1_pose)

    # The matrix old_poses[i] would take a point from the i-th camera 
    # coordinate system to the old world (W1) coordinate system. The 
    # matrix W1_to_W2_pose would take a point from the old world 
    # coordinate system (W1) to the new world coordinate system (W2). 
    # Hence, the matrix new_poses[i] would take a point from the i-th 
    # camera coordinate system to the new world world coordinate 
    # system (W2). 
    temp_poses = W1_to_W2_pose @ old_poses

    # Scaling poses and bounds so that ... (TODO: Elaborate).
    new_poses, new_bounds = scale_poses_and_bounds(
        poses = temp_poses, bounds = old_bounds
    )

    return new_poses, new_bounds

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
        method      :   A string which is either "average" 
                        or "min_dist"

    Returns:
        origin      :   A NumPy array of shape (3,) which denotes 
                        the location of the new world world coordinate 
                        system (W2).
    """
    if method == "average":
        cam_locs = poses[:, :3, 3]
        origin = np.mean(cam_locs, axis = 0)

    elif method == "min_dist":
        raise NotImplementedError(
            "The method min_dist is not implemented yet."
        )
    
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
    
    # Getting the average X-Axis direction of the cameras and the average 
    # Y-Axis direction of the cameras to serve as two reference vectors.
    avg_x_direction = np.mean(poses[:, :3, 0], axis = 0)
    avg_y_direction = np.mean(poses[:, :3, 1], axis = 0)

    # Setting the y_basis of the new new world coordinate 
    # system (W2) as avg_y_direction
    y_basis = avg_y_direction

    # Using avg_x_direction and y_basis to get z_basis. Note that 
    # z_basis is perpendicular to avg_x_direction and y_basis.
    z_basis = np.cross(avg_x_direction, y_basis)

    # Using y_basis and z_basis to get x_basis. Note that 
    # x_basis, y_basis and z_basis are now mutually perpendicular.
    x_basis = np.cross(y_basis, z_basis)

    return x_basis, y_basis, z_basis

def scale_poses_and_bounds(old_poses, old_bounds):
    """
    TODO: Elaborate.
    
    Rough idea is that we need to move from the W2 coordinate system to 
    the SW2 coordinate system. In this coordinate system, the XYZ points 
    along the central ray for all the cameras will always lie within [-1, 1]. 

    TODO: Should this function ensure this only for the central ray and add 
    extra scaling, or consider the rays along each camera frustum as well?
    """
    raise NotImplementedError("Need to implement.")

    return new_poses, new_bounds

def scale_imgs_and_intrinsics(old_imgs, old_intrinsics, scale_factor):
    """
    Scales images and intrinsics by the given scaling factor.

    The scale factor can be a single float or a tuple of two floats.

    TODO: Elaborate.
    """
    if type(scale_factor) is tuple:
        sx, sy = scale_factor
        assert (type(sx) is float) and (type(sy) is float)
    
    elif type(scale_factor) is float:
        sx, sy = scale_factor, scale_factor

    else:
        raise ValueError("Invalid type of scale_factor.")

    raise NotImplementedError("Need to implement.")

    return new_imgs, new_intrinsics
