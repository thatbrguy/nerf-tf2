import cv2
import numpy as np
import tensorflow as tf

def make_4x4(arr):
    """
    Converts a 3x4 matrix into a 4x4 matrix. The last row of
    the 4x4 matrix will be [0, 0, 0, 1].

    Args:
        arr     :   A NumPy array of shape (3, 4)

    Returns:
        output  :   A NumPy array of shape (4, 4)
    """
    assert arr.shape == (3, 4)

    # The last row of np.eye(4) is [0, 0, 0, 1]. Hence, we 
    # create np.eye(4) and overwrite the first three rows of 
    # it with arr to get our desired output.
    output = np.eye(4)
    output[:3, :] = arr

    return output

def make_homogeneous(points):
    """
    Converts a points array of shape (N, 3) to (N, 4). The 
    last column will have ones.

    Args:
        points  :   A NumPy array of shape (N, 3)

    Returns:
        output  :   A NumPy array of shape (N, 4)
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
    """
    assert len(vec.shape) <= 2
    EPS = 1e-8
    
    if len(vec.shape) == 1:
        magnitude = np.sqrt(np.sum(vec ** 2))
        norm_vec = vec / (magnitude + EPS)

    elif len(vec.shape) == 2:
        magnitude = np.sqrt(np.sum(vec ** 2, axis = 1))
        norm_vec = vec / (magnitude[:, None] + EPS)

    return norm_vec

def rotate_vectors(arr, vectors):
    """
    Rotates the given 3D vectors using the rotation matrix 
    extracted from arr.

    Args:
        arr     :   A NumPy array of shape (3, 3) or (3, 4) 
                    or (4, 4). In each case arr[:3, :3] must 
                    contain the rotation matrix.
        vectors :   A NumPy array of shape (N, 3). Here, N is 
                    the number of vectors.

    Returns:
        output  :   A NumPy array of shape (N, 3). Here, N is 
                    the number of vectors.
    """
    check_1 = arr.shape == (3, 3)
    check_2 = arr.shape == (3, 4)
    check_3 = arr.shape == (4, 4)

    assert np.any([check_1, check_2, check_3]), (
        "Shape of arr is invalid. Must be "
        "either (3, 3), (3, 4) or (4, 4)"
    )
    assert vectors.shape[1] == 3

    transform = arr[:3, :3]
    output = (transform @ vectors.T).T

    return output

def transform_points(arr, points):
    """
    Applies a given transformation to the points.

    Args:
        arr     :   A NumPy array of shape (4, 4) or (3, 4).
        points  :   A NumPy array of shape (N, 3). Here, N is 
                    the number of points.

    Returns:
        output  :   A NumPy array of shape (N, 3) with the 
                    transformed points. Here, N is the number 
                    of points.
    """
    assert points.shape[1] == 3

    if arr.shape == (3, 4):
        transform = make_4x4(arr)
    elif arr.shape == (4, 4):
        transform = arr
    else:
        raise ValueError(
            "Shape of arr is invalid. Must be either (3, 4) or (4, 4)"
        )

    homogeneous_pts = make_homogeneous(points)
    output = (transform @ homogeneous_pts.T).T

    # Removing the last column which will be all ones.
    output = output[:, :3]

    return output

def batched_transform_points(arrs, points):
    """
    Applies a batch of transformations to the given points.

    The variable arrs should contain M transformation matrices. 
    Hence, shape of arrs should be (M, 4, 4). Each transformation
    matrix is applied to all the points.
    
    TODO: verify shapes.

    Args:
        arrs    :   A NumPy array of shape (M, 4, 4). Here, M
                    is the number of transformation matrices.
        points  :   A NumPy array of shape (N, 3). Here, N is 
                    the number of points.
    
    Returns:
        output  :   A NumPy array of shape (M, N, 3)
    """
    assert arrs.shape[1:] == (4, 4), (
        "Shape of arrs is invalid. Must be (M, 4, 4)"
    )
    assert points.shape[1] == 3

    homogeneous_pts = make_homogeneous(points)
    output = (arrs @ homogeneous_pts.T).transpose(0, 2, 1)

    # Removing the last column which will be all ones.
    output = output[..., :3]

    return output

def transform_line_segments(arr, lines):
    """
    Transforms line segments.

    The array lines is a collection of line segments with lines[i]
    containing the i-th line segment. Each line segment is comprised
    of its two end points. This function applies a transformation
    to the points of each line segment.

    TODO: verify shapes.
    
    Args:
        arr         :   A NumPy array of shape (4, 4) or (3, 4).
        lines       :   A NumPy array of shape (N, 2, 3). Here, N is 
                        the number of line segments.

    Returns:
        output      :   A NumPy array of shape (N, 2, 3) with the 
                        transformed line segments. Here, N is the 
                        number of line segments.
    """
    assert (lines.shape[1] == 2) and (lines.shape[2] == 3)
    N = lines.shape[0]
    points = lines.reshape(-1, 3)

    out_points = transform_points(arr, points)
    output = out_points.reshape(N, 2, 3)

    return output

def batched_transform_line_segments(arrs, lines):
    """
    Applies a batch of transformations to the given line segments.

    The array lines is a collection of line segments with lines[i]
    containing the i-th line segment. Each line segment is comprised
    of its two end points. This function applies a batch of 
    transformations to the points of each line segment.

    TODO: verify shapes.
    
    Args:
        arrs        :   A NumPy array of shape (M, 4, 4). Here, M
                        is the number of transformation matrices.
        lines       :   A NumPy array of shape (N, 2, 3)

    Returns:
        output      :   A NumPy array of shape (M, N, 2, 3)
    """
    assert (lines.shape[1] == 2) and (lines.shape[2] == 3)
    M, N = arrs.shape[0], lines.shape[0]
    points = lines.reshape(-1, 3)

    out_points = batched_transform_points(arrs, points)
    output = out_points.reshape(M, N, 2, 3)

    return output

def calculate_new_world_transform(poses, origin_method, basis_method, manual_rotation = None):
    """
    Computes the transformation between W1 coordinate system and 
    the W2 coordinate system.

    Given N pose matrices (poses), each of which can transform a
    point from a camera coordinate system to the W1 world coordinate
    system, this function configures a new world coordinate system (W2). 
    This function computes and returns the transformation needed to go
    from the W1 coordinate system to the W2 coordinate system.

    For more information about the various coordinate systems, please
    refer to the documentation.

    Args:
        poses                   :   A NumPy array of shape (N, 4, 4)
        origin_method           :   A string which is either "average", 
                                    "min_dist_solve" or "min_dist_opt"
        basis_method            :   A string which is either "identity",
                                    "compute" or "manual".
        manual_rotation         :   This parameter can take a string 
                                    or None. If basis_method is "manual", 
                                    then this parameter must be a string 
                                    and this parameter must be a path that 
                                    points to a NumPy file which contains 
                                    a rotation matrix of shape (3, 3). 
                                    If basis_method is "identity" or "compute", 
                                    then this parameter is not used and can 
                                    be left as None.

    Returns:
        W1_to_W2_transform       :   A NumPy array of shape (4, 4)
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

    elif basis_method == "manual":
        assert manual_rotation is not None
        matrix = np.load(manual_rotation)

        assert matrix.shape == (3, 3)
        x_basis, y_basis, z_basis = matrix[:, 0], matrix[:, 1], matrix[:, 2]

    else:
        raise ValueError(f"Invalid basis_method: {basis_method}")

    W2_to_W1_3x4 = np.stack([x_basis, y_basis, z_basis, origin], axis = 1)
    W2_to_W1_transform = make_4x4(W2_to_W1_3x4)
    W1_to_W2_transform = np.linalg.inv(W2_to_W1_transform)
    
    return W1_to_W2_transform

def calculate_scene_scale(
        poses, bounds, bounds_method, 
        intrinsics = None, height = None, width = None
    ):
    """
    Calculates a scale factor for the 360 inward facing scene so that 
    the coordinates of the XYZ points that are to be given to the neural 
    networks can lie within the range [-1, 1].

    We would like to constuct a new coordinate system W3 such that the 
    XYZ points in the new W3 coordinate system which are of interest to us
    "attempts" to satisfy a constraint. By "of interest to us" we refer 
    to only the XYZ points in the W3 coordinate system which are to be 
    given as input to the neural networks.

    The constraint is that the X, Y, Z components of any point in the 
    W3 coordinate system that is of interest to us should be within
    the range [-1, 1]. This is a requirement for the positional 
    encoding layer.

    We could "attempt" to satisfy this constraint by simply scaling
    the W2 coordinate system. Hence, we would like to create a new 
    coordinate system W3 which is just a scaled version of the W2
    coordinate system. This function attempts to calculate just the
    scale factor.

    The key word to note is "attempts" to satisfy the constraint. 
    In practice, it would be better to further adjust the calculated
    scale factor to be careful. To adjust the calculated scale factor,
    some hyperparameters are available in the config file. This 
    function DOES NOT perform the adjustment. This adjustment is performed
    in the function _reconfigure_scene_scale in the class Dataset in the file 
    nerf/core/base_dataset.py.

    Please read through the code and comments in this function to understand 
    the exact methods used to calculate the scale factor. 
    
    Args:
        poses               :   A NumPy array of shape (N, 4, 4) denoting the
                                camera to W2 transformation matrices.
        bounds              :   A NumPy array of shape (N, 2).
        bounds_method       :   A string which is either "include_corners" 
                                or "central_ray"
        intrinsics          :   A NumPy array of shape (N, 3, 3)
        height              :   An integer denoting the height of the images.
        width               :   An integer denoting the width of the images.


    Returns:
        scene_scale_factor  :   The computed scene scale factor.
    """
    rays_o = poses[:, :3, 3]
    rays_d = poses[:, :3, 2]
    far = bounds[:, 1]

    points_far = rays_o + far[:, None] * rays_d

    # We would like the X, Y, Z components of each point in the array points
    # to be within the range [-1, 1] in the W3 coodinate system. Do note that
    # the array points contains points which are in the W2 coordinate system.
    # The bounds_method controls what kind of points are to be included in
    # the points array.
    if bounds_method == "include_corners":
        check_1 = intrinsics is not None
        check_2 = height is not None
        check_3 = width is not None
        assert check_1 and check_2 and check_3
        
        corner_points = get_corner_ray_points(
            poses = poses, bounds = bounds, 
            intrinsics = intrinsics, height = height, width = width
        )
        points = np.concatenate([rays_o, points_far, corner_points], axis = 0)

    elif bounds_method == "central_ray":
        points = np.concatenate([rays_o, points_far], axis = 0)

    else:
        raise ValueError(f"Invalid bounds_method: {bounds_method}")

    # To find the scale factor, first we find the largest absolute X, Y and Z
    # components among all the points. Then, we select largest value (let's call
    # this largest value largest_proj) among the largest absolute X, Y and Z components.
    # The reciprocal of largest_proj gives us the scale factor.
    projs = np.abs(points).max(axis = 0)
    largest_proj = projs.max()
    scene_scale_factor = 1 / largest_proj

    # For more intuition, imagine a cube whose centroid is at the origin of the 
    # W2 coordinate system, and whose side length is 2*largest_proj. We say that
    # all points in the points array lie within/on this cube. Each component of any 
    # of this cube's vertices would be +largest_proj or -largest_proj. Now, imagine
    # scaling the coordinate system such that the side length of the cube is now 2*1.
    # Each component of any of this cube's vertices in the scaled coordinate system 
    # would be +1 or -1. But do not that our points are within the cube. Hence, we 
    # "attempted" to satisfy the constraint that is mentioned in the docstring! 
    # And hence, 1/largest_proj is the required scaling factor.

    # The word "attempted" is used above because we are only using the array points 
    # to calculate largest_proj. We hope that those points are sufficient to calculate 
    # a good scale factor. But if they are not sufficient, the constraint may not be 
    # satisfied. For our use case, just the points array should be mostly sufficient. 
    # For safety though, it is recommended to adjust the scale factor by setting 
    # the necessary hyperparameters in the config yaml file. The adjusted is done
    # elsewhere in the code.

    return scene_scale_factor

def reconfigure_poses(old_poses, W1_to_W2_transform):
    """
    Reconfigures the poses.

    TODO: Updated description to mention this function works for 
    both batched and non-batched modes

    The matrix old_poses[i] would take a point from the i-th camera 
    coordinate system to the old world (W1) coordinate system. The 
    matrix W1_to_W2_transform would take a point from the old world 
    coordinate system (W1) to the new world coordinate system (W2). 
    Hence, the matrix new_poses[i] would take a point from the i-th 
    camera coordinate system to the new world coordinate system (W2). 
    
    Args:
        old_poses           : A NumPy array of shape (N, 4, 4) or (4, 4)
        W1_to_W2_transform  : A NumPy array of shape (4, 4)


    Returns:
        new_poses           :   A NumPy array of shape (N, 4, 4) or (4, 4)
    """
    new_poses = W1_to_W2_transform @ old_poses
    return new_poses

def reconfigure_scene_scale(old_poses, old_bounds, scene_scale_factor):
    """
    Reconfigures the scene scale.

    TODO: Updated description to mention this function works for 
    both batched and non-batched modes

    Given a scale factor, this function calculates new poses and bounds.
    The old poses are assumed to be camera to W2 transformation matrices.
    The new poses will be camera to W3 transformation matrices. Do note
    that the W3 coordinate system is just a scaled version of the
    W2 coordinate system. The new bounds are also calculated.

    If the scale factor is >= 1, then the new poses and new bounds are
    same as the old poses and old bounds respectively.

    Args:
        old_poses           :   A NumPy array of shape (N, 4, 4) or (4, 4)
        old_bounds          :   A NumPy array of shape (N, 2) or (2,)
        scene_scale_factor  :   A TODO type denoting the scale factor of
                                the scene.

    Returns:
        new_poses           :   A NumPy array of shape (N, 4, 4) or (4, 4)
    """
    if scene_scale_factor >= 1:
        # No scaling required for this case.
        new_poses = old_poses
        new_bounds = old_bounds

    elif scene_scale_factor < 1:
        # Poses and bounds are reconfigured in this case.
        scale_transform = np.eye(4) * scene_scale_factor
        scale_transform[3, 3] = 1

        new_poses = scale_transform @ old_poses.copy()
        new_bounds = old_bounds.copy() * scene_scale_factor

    return new_poses, new_bounds

def scale_imgs_and_intrinsics(old_imgs, old_intrinsics, scale_factor):
    """
    Scales images and intrinsics by the given scaling factor.
    
    The user can choose to optionally scale the images by a scale factor
    for training purposes. If the images are scaled, then the intrinsic
    matrices also have to be adjusted.

    Args:
        old_imgs        :   A NumPy array of shape (N, H, W, 3)
        old_intrinsics  :   A NumPy array of shape (N, 3, 3)
        scale_factor    :   Can take None or a float value. If None is
                            provided, no scaling is formed. If a float
                            value is provided, scaling is performed.

    Returns:
        new_imgs        :   A NumPy array of shape (N, H, W, 3)
        new_intrinsics  :   A NumPy array of shape (N, 3, 3)
    """
    if scale_factor is None:
        # No scaling is performed in this case.
        new_imgs = old_imgs
        new_intrinsics = old_intrinsics

    elif scale_factor is not None:
        # Scaling is performed in this case.
        sx, sy = scale_factor, scale_factor
        new_imgs, new_intrinsics = [], []

        # TODO: Maybe add arg to choose interpolation?
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

            new_imgs.append(new_img)
            new_intrinsics.append(new_intrinsic)

        new_intrinsics = np.array(new_intrinsics)
        new_imgs = np.array(new_imgs)

    else:
        raise ValueError("Invalid setting of scale_factor.")

    return new_imgs, new_intrinsics

def get_corner_ray_points(poses, bounds, intrinsics, height, width):
    """
    Given N cameras, we get N corresponding images. Each image has
    4 corners. This function computes the corresponding XYZ points
    in the W2 frame for each corner for each image.

    The poses must be camera to W2 transformation matrices.
    TODO: Verify types

    Args:
        poses       :   A NumPy array of shape (N, 4, 4)
        bounds      :   A NumPy array of shape (N, 2)
        intrinsics  :   A NumPy array of shape (N, 3, 3)
        height      :   An integer denoting the height of the images.
        width       :   An integer denoting the width of the images.

    Returns:
        output      :   A NumPy array of shape (N * 4, 3) 
    """
    H, W = height, width

    # Shape of rays_o: (N, 3)
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
    
    IMPORTANT NOTE:
        This function is just a prototype. It is highly recommended to 
        use the function solve_min_dist_point instead.
    
    TODO: Verify shapes

    Args:
        poses   :   A NumPy array of shape (N, 4, 4)

    Returns:
        output  :   A NumPy array of shape (3,)
    """
    # Initializing the point with all zeros.
    point = tf.Variable([[0., 0., 0.]], dtype = tf.float64)
    
    # Getting the origin point of each line and the normalized 
    # direction vector of each line.
    origins = poses[:, :3, 3]
    directions = normalize(poses[:, :3, 2])

    # Converting origins and directions to TF constants.
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
            
            # loss is the sum of the squared distance to each line.
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
    
    A blog post explaining the theory will be written in the future.

    Reference:
        https://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines

    Args:
        poses   :   A NumPy array of shape (N, 4, 4)

    Returns:
        output  :   A NumPy array of shape (3,)
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
    Computes the origin of the new world coordinate system (W2) given 
    N pose matrices.

    Each of the N pose matrices can transform a point from a camera 
    coordinate system to a arbitrary world coordinate system (W1). 
    In this function, we want to compute the origin of a new 
    world coordinate system (W2).

    The origin of the new world coordinate system (W2) is represented
    in the W1 coordinate system.

    The argument "method" describes the method using which the origin
    is computed. The recommended method is "min_dist_solve"

    Args:
        poses       :   A NumPy array of shape (N, 4, 4)
        method      :   A string which is either "average", 
                        "min_dist_solve" or "min_dist_opt"

    Returns:
        origin      :   A NumPy array of shape (3,) which denotes 
                        the location of the origin of the new world 
                        coordinate system (W2) in the W1 coordinate
                        system.
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
    Computes the basis vectors of the new world coordinate system (W2). 
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

def create_spherical_path(radius, inclination, num_cameras, manual_rotation):
    """
    Computes a sequence of poses that trace a path on a sphere.

    The poses are camera to world transformation matrices. It is up to the
    user to define what "world" coordinate system means here. This function 
    assumes that a sphere of the specified radius is centered at the world
    origin. The poses are created such that the Z-Axis of the camera 
    coordinate systems point towards the world origin.

    The argument manual_rotation can take None or a string. If a string
    is provided, then it must be a path to a NumPy file which contains 
    a rotation matrix. In that case, the NumPy array is loaded, and then 
    the rotation is applied to the calculated poses. This can enable the
    user to control the path traced by the poses. If manual_rotation 
    is None, then no additional rotation is applied (the calculated 
    poses are returned as such)

    Further elaboration would be provided in the future.

    Args:
        radius          :   A float denoting the radius of the sphere.
        inclination     :   A float denoting the inclination in degrees.
        num_cameras     :   An integer denoting the number of cameras.
        manual_rotation :   Can take None or or a string. Please refer
                            to the docstring for more information.

    Returns:
        poses_4x4       :   A NumPy array of shape (N, 4, 4)
    """
    azimuth = np.linspace(0, 360, num_cameras, endpoint=False, dtype=np.float64)
    radius = np.full_like(azimuth, radius)
    inclination = np.full_like(azimuth, inclination)

    azimuth = np.radians(azimuth)
    inclination = np.radians(inclination)

    point_x = radius * np.sin(inclination) * np.cos(azimuth)
    point_y = radius * np.sin(inclination) * np.sin(azimuth)
    point_z = radius * np.cos(inclination)

    cam_origins = np.stack([point_x, point_y, point_z], axis=1)

    z_vec = -1 * cam_origins
    z_basis = normalize(z_vec)

    temp_x = -1 * radius * np.sin(inclination) * np.sin(azimuth)
    temp_y = radius * np.sin(inclination) * np.cos(azimuth)
    temp_z = np.zeros_like(temp_x)

    x_vec = np.stack([temp_x, temp_y, temp_z], axis = 1)
    x_basis = normalize(x_vec)

    y_vec = np.cross(z_basis, x_basis)
    y_basis = normalize(y_vec)

    poses = np.stack([x_basis, y_basis, z_basis, cam_origins], axis = -1)
    
    last_row = np.zeros((poses.shape[0], 1, 4), dtype = np.float64)
    poses_4x4 = np.concatenate([poses, last_row], axis = 1)
    poses_4x4[:, 3, 3] = 1.0

    if manual_rotation is not None:
        transform = np.eye(4, dtype = np.float64)
        
        matrix = np.load(manual_rotation)
        transform[:3, :3] = matrix

        poses_4x4 = transform @ poses_4x4

    return poses_4x4
