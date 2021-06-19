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
