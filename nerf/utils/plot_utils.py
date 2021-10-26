import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from nerf.utils import pose_utils

def create_eye_frustum_for_plots(length, width, height):
    """
    TODO: Docstring
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

def create_xyz_axes_for_plots(length):
    """
    TODO: Docstring
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
        c2w_matrices, plt_ax, plot_eye_frustums = True, 
        plot_cam_axes = True, frustum_color = "red",
    ):
    """
    Returns a 3D plot of the camera poses and other details.

    TODO: Make length, width, height tunable!
    """
    
    # Creating lines that constitute an eye frustum.
    eye_frustum = create_eye_frustum_for_plots(0.4, 0.4, 0.6) 

    # Creating lines that constitute an XYZ axes.
    camera_axes = create_xyz_axes_for_plots(0.3)

    # Moving the lines that constitute the eye frustum for each 
    # camera to the desired world frame.
    eyes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, eye_frustum
    )

    # Moving the lines that constitute the XYZ axes for each 
    # camera to the desired world frame.
    cam_axes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, camera_axes
    )

    for eye, axes in zip(eyes_world_frame, cam_axes_world_frame):

        # Plotting the lines that constitute the eye frustum.
        if plot_eye_frustums:
            for line in eye:
                plt_ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color = frustum_color)

        if plot_cam_axes:
            # Plotting the XYZ axes.
            line = plt_ax.plot3D(axes[0, :, 0], axes[0, :, 1], axes[0, :, 2], color = 'red')
            plt_ax.plot3D(axes[1, :, 0], axes[1, :, 1], axes[1, :, 2], color = 'green')
            plt_ax.plot3D(axes[2, :, 0], axes[2, :, 1], axes[2, :, 2], color = 'blue')
            plt_ax.scatter3D(axes[0,0,0], axes[0,0,1], axes[0,0,2], s = 1.0, c = "black")

def plot_scene(
    plot_gt = False, plot_inference = False, gt_poses = None, 
    inference_poses = None, W1_to_W2_transform = None, 
    plot_eye_frustums = True, plot_cam_axes = True, 
):
    """
    TODO: Docstring.
    """
    if plot_gt:
        assert gt_poses is not None

    if plot_inference:
        assert inference_poses is not None
        assert W1_to_W2_transform is not None

    if W1_to_W2_transform is not None:
        # inference_poses are assumed to be already in the W2 coordinate system.
        gt_poses = pose_utils.reconfigure_poses(
            gt_poses, W1_to_W2_transform
        )

    ## TODO: Keep using ortho?
    ax = plt.axes(projection = '3d', proj_type = 'ortho')
    
    if plot_gt:
        _plot_cameras(
            gt_poses, ax, plot_eye_frustums = plot_eye_frustums, 
            plot_cam_axes = plot_cam_axes, frustum_color = "red",
        )

    if plot_inference:
        _plot_cameras(
            inference_poses, ax, plot_eye_frustums = plot_eye_frustums, 
            plot_cam_axes = plot_cam_axes, frustum_color = "green"
        )

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
