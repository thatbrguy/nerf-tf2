import os
import logging
import argparse
import numpy as np
import tensorflow as tf

from nerf.core import datasets
from nerf.utils import pose_utils, plot_utils
from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(
    logger, plot_gt, plot_inference, splits = None, 
    coord_system = "W2", plot_eye_pyramids = True, plot_cam_axes = True
):
    """
    Launches the visualization run.
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    assert type(plot_gt) is bool
    assert type(plot_inference) is bool

    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)

    # Getting data
    data_splits, num_imgs, dataset_obj =\
        datasets.get_data_and_metadata_for_splits(params, return_dataset_obj=True)

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
            manual_rotation = render_params.manual_rotation,
        )

    else:
        inference_poses = None

    if coord_system == "W2":
        W1_to_W2_transform, _ = dataset_obj.load_reconfig_params()
    elif coord_system == "W1":
        W1_to_W2_transform = None
    else:
        raise ValueError(f"Invalid setting of coord_system: {coord_system}")

    plot_utils.plot_scene(
        plot_gt = plot_gt, plot_inference = plot_inference, gt_poses = gt_poses, 
        inference_poses = inference_poses, W1_to_W2_transform = W1_to_W2_transform, 
        plot_eye_pyramids = plot_eye_pyramids, plot_cam_axes = plot_cam_axes, 
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
    MPL_logger = logging.getLogger("matplotlib")
    MPL_logger.setLevel(logging.WARNING)

    launch(
        logger, plot_gt = True, plot_inference = True, coord_system = "W2",
        splits = ["train", "val", "test"]
    )
