import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from nerf.utils import pose_utils

def _create_eye_pyramid_for_plots(length, width, height):
    """
    Calculates the line segments of a pyramid with a rectangular base in 3D space 
    so that it can be used to represent a camera in plots.
    
    ========
    Overview
    ========
    
    This function computes all the line segments that are necessary to 
    plot the pyramid.
    
    A single line segment is represented by two end points. Each end point 
    is a point in 3D space. The array lines contains multiple line segments. 
    Each row in lines (example: lines[0]) represents one line segment.

    All points described here are to be assumed in a camera coordinate system. 
    These points can be transformed a different coordinate system before plotting.
    
    ===================
    Pyramid Description
    ===================

    Let the centroid of the rectangular base of the pyramid be the point M. Let 
    the origin be the point O. The apex/tip of the pyramid is set to the origin O. 
    The direction vector OM (where OM = M - O) is a vector along the height of 
    the pyramid. The pyramid is oriented such that the vector OM is along the Z-Axis. 
    This pyramid can be used in plots to represent the viewing direction of a camera.

    ====================
    Function Description
    ====================

    Args:
        length  :   A float denoting the length of the rectangular base of the pyramid.
        width   :   A float denoting the width of the rectangular base of the pyramid.
        height  :   A float denoting the height of the pyramid.

    Returns:
        lines   :   A NumPy array of shape (8, 3) with the end points of the 
                    line segments that represents all the edges of the pyramid.

    """
    L, W, H = length, width, height

    O = np.array([0., 0., 0.], dtype = np.float64)
    A = np.array([-L/2, -W/2, H], dtype = np.float64)
    B = np.array([-L/2,  W/2, H], dtype = np.float64)
    C = np.array([ L/2,  W/2, H], dtype = np.float64)
    D = np.array([ L/2, -W/2, H], dtype = np.float64)

    lines = np.array([
        [O, A], [O, B], [O, C], [O, D],
        [A, B], [B, C], [C, D], [D, A],
    ])

    return lines

def _create_xyz_axes_for_plots(length):
    """
    Calculates the line segments that can be used to draw XYZ axes in plots.

    ========
    Overview
    ========
    
    This function computes all the line segments that are necessary to 
    draw XYZ axes in plots.
    
    A single line segment is represented by two end points. Each end point 
    is a point in 3D space. The array lines contains multiple line segments. 
    Each row in lines (example: lines[0]) represents one line segment.

    All points described here are to be assumed in a camera coordinate system. 
    These points can be transformed a different coordinate system before plotting.

    ====================
    Function Description
    ====================

    Args:
        length  :   A float denoting the length of an axis.

    Returns:
        lines   :   A NumPy array of shape (3, 3) with the end points of the 
                    line segments that represents the XYZ axes.
    """
    L = length

    O = np.array([0., 0., 0.], dtype = np.float64)
    X = np.array([  L,  0.,  0.], dtype = np.float64)
    Y = np.array([ 0.,   L,  0.], dtype = np.float64)
    Z = np.array([ 0.,  0.,   L], dtype = np.float64)

    lines = np.array([
        [O, X], [O, Y], [O, Z],
    ])

    return lines

def _plot_cameras(
    c2w_matrices, plt_ax, plot_eye_pyramids = True, 
    plot_cam_axes = True, pyramid_color = "red",
):
    """
    Plots cameras.

    Given N camera to world transformation matrices (c2w_matrices) representing 
    the pose of N cameras, this function plots the N cameras in the world frame.

    A camera can be plotted in two ways. The first method is to plot an "eye pyramid", 
    and the second method is to plot the "cameras axes". The user can choose to 
    plot both representations or just choose to plot just one representation.

    The "eye pyramid" method plots a pyramid such that the apex/tip of the pyramid is 
    at the location of the camera in the world coordinate system, and the height vector 
    of the pyramid is along the Z-Axis direction of the camera. For more information,
    about the pyramid orientation, you can refer to the function _create_eye_pyramid_for_plots

    The "cameras axes" method plots three line segments and a point. The line segment 
    which is plotted in red color represents the X-Axis direction of the camera as seen 
    in the world world coordinate system. The line segment which is plotted in green color 
    represents the Y-Axis direction of the camera as seen in the world world coordinate
    system. The line segment which is plotted in blue color represents the Z-Axis direction
    of the camera as seen in the world world coordinate system. The point which is plotted
    in black color (TODO: consider diff color?) represents the location of the camera in 
    the world coordinate system.

    Args:
        c2w_matrices        :   A NumPy array of shape (N, 4, 4) (TODO:verify) denoting 
                                the camera to world transformation matrices of N cameras.
        plt_ax              :   TODO; explain.
        plot_eye_pyramids   :   A boolean which can enable/disable plotting the eye 
                                pyramids. Default is set to True.
        plot_cam_axes       :   A boolean which can enable/disable plotting the camera
                                axes. Default is set to True.
        pyramid_color       :   Color of the eye pyramids. Default is set to "red".
    """
    # Creating lines that constitute an eye pyramid.
    eye_pyramid = _create_eye_pyramid_for_plots(0.4, 0.4, 0.6) 

    # Moving the lines that constitute the eye pyramid for each 
    # camera to the desired world frame.
    eyes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, eye_pyramid
    )
        
    # Creating lines that constitute an XYZ axes.
    camera_axes = _create_xyz_axes_for_plots(0.3)

    # Moving the lines that constitute the XYZ axes for each 
    # camera to the desired world frame.
    cam_axes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, camera_axes
    )

    for eye, axes in zip(eyes_world_frame, cam_axes_world_frame):

        # Plotting the lines that constitute the eye pyramid.
        if plot_eye_pyramids:
            for line in eye:
                plt_ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color = pyramid_color)

        if plot_cam_axes:
            # Plotting the XYZ axes.
            plt_ax.plot3D(axes[0, :, 0], axes[0, :, 1], axes[0, :, 2], color = 'red')
            plt_ax.plot3D(axes[1, :, 0], axes[1, :, 1], axes[1, :, 2], color = 'green')
            plt_ax.plot3D(axes[2, :, 0], axes[2, :, 1], axes[2, :, 2], color = 'blue')
            plt_ax.scatter3D(axes[0,0,0], axes[0,0,1], axes[0,0,2], s = 1.0, c = "black")

def plot_scene(
    plot_gt = False, plot_inference = False, gt_poses = None, 
    inference_poses = None, W1_to_W2_transform = None, 
    plot_eye_pyramids = True, plot_cam_axes = True, 
):
    """
    Plots the elements of the scene.

    This function plots the elements of the scene that the user has configured. 
    More specifically, this function can plot the ground truth cameras and/or 
    the inference cameras in the scene.

    The scene can be configured to be either in the W1 coordinate system or 
    the W2 coordinate system. 

    If W1_to_W2_transform is None, then the scene is configured to be in 
    the W1 coordinate system. 

    If W1_to_W2_transform is not None, then a NumPy array of shape (4, 4) 
    which can transform a point from the W1 corodinate system to the 
    W2 coordinate system must be provided. In this case, the scene is 
    configured to be in the W2 coordinate system. 

    If inference cameras are to be plotted, then only the W2 coodinate 
    system can be used. For more information about the W1 and W2 coordinate 
    systems, please refer to the coordinate systems documentation.

    Args:
        plot_gt             :   A boolean which can enable/disable plotting cameras 
                                mentioned by the ground truth poses (gt_poes). 
                                Default is set to True.
        plot_inference      :   A boolean which can enable/disable plotting cameras 
                                belonging to the inference poses (inference_poses). 
                                Default is set to True.
        gt_poses            :   A NumPy array of shape (N, 4, 4) (TODO:verify) with 
                                the poses of the N ground truth cameras.
        inference_poses     :   A NumPy array of shape (M, 4, 4) (TODO:verify) with 
                                the poses of the M inference cameras.
        W1_to_W2_transform  :   This argument can take None or a NumPy array of 
                                shape (4, 4). Please refer to the docstring for 
                                more information on how to set this argument.
        plot_eye_pyramids   :   A boolean which can enable/disable plotting the eye 
                                pyramids for every camera. Default is set to True. 
                                Please refer to the docstring of _plot_cameras for 
                                more information.
        plot_cam_axes       :   A boolean which can enable/disable plotting the eye 
                                pyramids for every camera. Default is set to True. 
                                Please refer to the docstring of _plot_cameras for 
                                more information.
    """
    if plot_gt:
        assert gt_poses is not None

    if plot_inference:
        assert inference_poses is not None
        assert W1_to_W2_transform is not None

    if (W1_to_W2_transform is not None) and (gt_poses is not None):
        # inference_poses are assumed to be already in the W2 coordinate system.
        gt_poses = pose_utils.reconfigure_poses(
            gt_poses, W1_to_W2_transform
        )

    ax = plt.axes(projection = '3d', proj_type = 'ortho')
    
    if plot_gt:
        _plot_cameras(
            gt_poses, ax, plot_eye_pyramids = plot_eye_pyramids, 
            plot_cam_axes = plot_cam_axes, pyramid_color = "red",
        )

    if plot_inference:
        _plot_cameras(
            inference_poses, ax, plot_eye_pyramids = plot_eye_pyramids, 
            plot_cam_axes = plot_cam_axes, pyramid_color = "green"
        )

    # Plotting the origin of the world coordinate system
    ax.scatter3D([0], [0], [0], s = 1.0, c = "black")

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
                
    plt.show()

if __name__ == "__main__":
    pass
