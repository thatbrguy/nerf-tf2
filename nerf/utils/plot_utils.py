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

def plot_scene(c2w_matrices, plot_eye_frustums = True, plot_cam_axes = True):
    """
    Returns a 3D plot of the camera poses and other details.

    TODO: Make length, width, height tunable!
    """
    
    # Creating lines that constitute an eye frustum.
    eye_frustum = create_eye_frustum_for_plots(0.4, 0.4, 0.6) 

    # Creating lines that constitute an XYZ axes.
    camera_axes = create_xyz_axes_for_plots(0.3)

    # Moving the lines that constitute the eye frustum for each 
    # camera to the world frame.
    eyes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, eye_frustum
    )

    # Moving the lines that constitute the XYZ axes for each 
    # camera to the world frame.
    cam_axes_world_frame = pose_utils.batched_transform_line_segments(
        c2w_matrices, camera_axes
    )

    ## TODO: Keep using ortho?
    ax = plt.axes(projection = '3d', proj_type = 'ortho')

    for eye, axes in zip(eyes_world_frame, cam_axes_world_frame):

        # Plotting the lines that constitute the eye frustum.
        if plot_eye_frustums:
            for line in eye:
                ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color = 'red')

        if plot_cam_axes:
            # Plotting the XYZ axes.
            ax.plot3D(axes[0, :, 0], axes[0, :, 1], axes[0, :, 2], color = 'red')
            ax.plot3D(axes[1, :, 0], axes[1, :, 1], axes[1, :, 2], color = 'green')
            ax.plot3D(axes[2, :, 0], axes[2, :, 1], axes[2, :, 2], color = 'blue')
            ax.scatter3D(axes[0,0,0], axes[0,0,1], axes[0,0,2], s = 1.0, c = "black")

    ax.scatter3D([0], [0], [0], s = 1.0, c = "black")
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
                
    plt.show()

if __name__ == "__main__":

    # plot_scene(c2w_matrices, plot_eye_frustums = True, plot_cam_axes = False)
