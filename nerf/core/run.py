import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from nerf.core import ops
from nerf.core.model import NeRF
from nerf.core.datasets import BlenderDataset

from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def setup_datasets(params):
    """
    Sets up the train and val datasets.
    """
    loader = BlenderDataset(params = params)
    tf_datasets, num_imgs, img_HW = loader.get_dataset()

    if params.data.advance_train_loader.enable:
        skip_count = params.data.advance_train_loader.skip_count
        tf_datasets["train"] = tf_datasets["train"].skip(skip_count)

    return tf_datasets, num_imgs, img_HW

def setup_model_and_callbacks(params, num_imgs, img_HW):
    """
    Sets up the NeRF model and the list of callbacks.
    """
    ## Setting up callbacks.
    saver = ops.CustomSaver(params = params, save_best_only = False)
    tensorboard = TensorBoard(
        log_dir = params.system.tensorboard_dir, 
        update_freq = "epoch"
    )
    callbacks = [saver, tensorboard]
    
    # Can use ops.LogValImages only if eager mode is enabled.
    if params.system.run_eagerly and params.system.log_images:
        val_imgs_logger = ops.LogValImages(
            params = params, height = img_HW[0], 
            width = img_HW[1], num_val_imgs = num_imgs["val"]
        )
        callbacks += [val_imgs_logger]

    ## Setting up the optimizer and LR scheduler.
    lr_schedule = ExponentialDecay(
        initial_learning_rate = 5e-4,
        decay_steps = 500000,
        decay_rate = 0.1,
    )
    optimizer = Adam(learning_rate = lr_schedule)
    psnr_metric = ops.PSNRMetric()

    # Instantiating the model.
    nerf = NeRF(params = params)

    # Enabling eager mode for ease of debugging. 
    nerf.compile(
        optimizer = optimizer, 
        metrics = [psnr_metric], 
        run_eagerly = params.system.run_eagerly,
    )

    if params.model.load.set_weights:
        nerf.set_everything()

    return nerf, callbacks

def launch(logger):
    """
    TODO: Docstring
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    # Setting TF seed to enable determinism of TF.
    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)
    
    # Getting datasets and specs
    tf_datasets, num_imgs, img_HW = setup_datasets(params)

    # Setting up some dataset related parameters.
    repeat_count = params.data.repeat_count
    # Total number of pixels when the entire dataset is repeated repeat_count times:
    total_pixels = (img_HW[0] * img_HW[1] * num_imgs["train"] * repeat_count)
    # Total number of steps:
    total_steps = int(total_pixels / params.data.batch_size)
    # Steps per epoch:
    steps_per_epoch = params.system.steps_per_epoch
    # Number of epochs:
    num_epochs = int(total_steps / steps_per_epoch)

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
