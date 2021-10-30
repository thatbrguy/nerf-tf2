import os
import logging
import numpy as np
import tensorflow as tf

from nerf.core import ops, datasets
from nerf.core.model import setup_model_and_callbacks
from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(logger):
    """
    Launches the training run.
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)
    
    # Getting datasets and specs
    tf_datasets, num_imgs, img_HW = \
        datasets.get_tf_datasets_and_metadata_for_splits(params)

    # Setting up some dataset related parameters.
    if params.data.dataset_mode == "iterate":
        repeat_count = params.data.iterate_mode.repeat_count
        # Total number of pixels when the entire dataset is repeated repeat_count times:
        total_pixels = (img_HW[0] * img_HW[1] * num_imgs["train"] * repeat_count)
        # Total number of steps:
        total_steps = int(total_pixels / params.data.batch_size)
        # Steps per epoch:
        steps_per_epoch = params.system.steps_per_epoch
        # Number of epochs:
        num_epochs = int(total_steps / steps_per_epoch)

    elif params.data.dataset_mode == "sample":
        repeat_count = params.data.sample_mode.repeat_count
        # Total number of steps:
        total_steps = int(num_imgs["train"] * repeat_count)
        # Steps per epoch:
        steps_per_epoch = params.system.steps_per_epoch
        # Number of epochs:
        num_epochs = int(total_steps / steps_per_epoch)

    else:
        raise ValueError(f"Invalid dataset mode: {params.data.dataset_mode}")

    logger.debug(f"Number of Steps: {total_steps}")
    logger.debug(f"Number of Epochs: {num_epochs}")
    
    # Setup model and callbacks.
    nerf, callbacks = setup_model_and_callbacks(params, num_imgs, img_HW)

    nerf.fit(
        x = tf_datasets["train"], epochs = num_epochs,
        validation_data = tf_datasets["val"],
        validation_freq = params.system.validation_freq,
        callbacks = callbacks,
        steps_per_epoch = steps_per_epoch,
        initial_epoch = params.system.initial_epoch,
    )

if __name__ ==  "__main__":

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

    launch(logger)
