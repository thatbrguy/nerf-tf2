import os
import logging
import numpy as np
import tensorflow as tf

from nerf.core.datasets import BlenderDataset, CustomDataset
from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def setup_datasets(params):
    """
    Sets up the datasets.
    """
    if params.system.dataset_type == "BlenderDataset":
        loader = BlenderDataset(params = params)
    elif params.system.dataset_type == "CustomDataset":
        loader = CustomDataset(params = params)
    else:
        raise ValueError(f"Invalid dataset type: {params.system.dataset_type}")

    tf_datasets, num_imgs, img_HW = loader.get_dataset()

    if (type(loader) is not BlenderDataset) and \
        (params.system.white_bg):
        raise AssertionError("white_bg is only supported for BlenderDataset")

    if (params.data.dataset_mode == "iterate") and \
        (params.data.iterate_mode.advance_train_loader.enable):

        skip_count = params.data.iterate_mode.advance_train_loader.skip_count
        tf_datasets["train"] = tf_datasets["train"].skip(skip_count)

    return tf_datasets, num_imgs, img_HW

if __name__ == "__main__":

    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)
    from nerf.utils import plot_utils, pose_utils

    # Setting TF seed to enable determinism of TF.
    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)

    c2w_matrices = pose_utils.create_spherical_path(
        radius = 5, inclination = 5, n_cameras = 20
    )
    plot_utils.plot_scene(c2w_matrices)
