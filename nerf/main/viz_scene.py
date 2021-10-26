import os
import logging
import argparse
import numpy as np
import tensorflow as tf

from nerf.utils import pose_utils, plot_utils
from nerf.core.datasets import get_data
from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(
    logger, plot_gt, plot_inference, splits = None, 
    coord_system = "W2", plot_eye_frustums = True, plot_cam_axes = True
):
    """
    TODO: Docstring
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)

    # Getting data
    data_splits, num_imgs, loader = get_data(params, return_dataset_obj=True)

    if plot_gt:
        assert (splits is not None) and (type(splits) is list)

        if len(splits) > 1:
            temp = [data_splits[x].poses for x in splits] 
            gt_poses = np.concatenate(temp, axis = 0)

        else:
            gt_poses = data_splits[splits[0]].poses

    else:
        gt_poses = None

    if plot_inference:

        render_params = params.render
        assert render_params.num_cameras > 0
        assert coord_system == "W2"

        # Please refer to the documentation for more information 
        # about how the poses for rendering is calculated.
        inference_poses = pose_utils.create_spherical_path(
            radius = render_params.radius, 
            num_cameras = render_params.num_cameras,
            inclination = render_params.inclination,
        )

    else:
        plot_inference = None

    if coord_system == "W2":
        W1_to_W2_transform, _ = loader.load_reconfig_params()
    elif coord_system == "W1":
        W1_to_W2_transform = None

    plot_utils.plot_scene(
        plot_gt = plot_gt, plot_inference = plot_inference, gt_poses = gt_poses, 
        inference_poses = inference_poses, W1_to_W2_transform = W1_to_W2_transform, 
        plot_eye_frustums = plot_eye_frustums, plot_cam_axes = plot_cam_axes, 
    )

if __name__ == "__main__":

    # Setting numpy print options for ease of debugging.
    np.set_printoptions(precision = 5, suppress = True)

    # Setting up logger.
    logging.basicConfig(
        format='[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG
    )
    logger = logging.getLogger()

    PIL_logger = logging.getLogger("PIL")
    PIL_logger.setLevel(logging.WARNING)

    launch(logger, plot_gt = True, plot_inference = True, splits = ["train", "val", "test"])
