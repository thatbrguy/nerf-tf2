import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from nerf.core import ops
from nerf.core.model import NeRF, NeRFLite
from nerf.core.dataset import CustomDataset

from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def setup_model_and_callbacks(params):
    """
    Sets up the NeRF model and the list of callbacks.
    """
    ## Setting up callbacks.
    model_ckpt = ops.CustomModelSaver(params = params, save_best_only = False)
    tensorboard = TensorBoard(
        log_dir = params.system.tensorboard_dir, 
        update_freq = "epoch"
    )
    callbacks = [model_ckpt, tensorboard]
    
    # Can use ops.LogValImages only if eager mode is enabled.
    if params.system.run_eagerly and params.system.log_images:
        val_imgs_logger = ops.LogValImages(params = params, val_spec = val_spec)
        callbacks += [val_imgs_logger]

    ## Setting up the optimizer and LR scheduler.
    lr_schedule = ExponentialDecay(
        initial_learning_rate = 5e-4,
        decay_steps = 200000,
        decay_rate = 0.1,
    )
    optimizer = Adam(learning_rate = lr_schedule)

    # Instantiating the model.
    nerf = NeRF(params = params)

    # Enabling eager mode for ease of debugging. 
    nerf.compile(
        optimizer = optimizer, 
        metrics = [ops.psnr_metric], 
        run_eagerly = params.system.run_eagerly,
    )

    if params.model.load.set_weights:
        nerf.set_everything()

    return nerf, callbacks

if __name__ ==  "__main__":

    # Setting numpy print options for ease of debugging.
    np.set_printoptions(precision = 5, suppress = True)
    
    # Setting up logger.
    logging.basicConfig(
        format='[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG
    )
    logger = logging.getLogger()
    
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    # Setting TF seed to enable determinism of TF.
    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)
    
    ## Setting up the dataset.
    loader = CustomDataset(params = params)
    train_dataset, val_dataset, train_spec, val_spec = loader.get_dataset()

    # Setting up some dataset related parameters.
    # Total number of pixels when the entire dataset is repeated 5 times:
    total_pixels = (800 * 800 * 100 * 5)
    # Total number of steps:
    total_steps = int(total_pixels / params.data.batch_size)
    # Steps per epoch:
    steps_per_epoch = params.system.steps_per_epoch
    # Number of epochs:
    num_epochs = int(total_steps / steps_per_epoch)

    logger.debug(f"Number of Steps: {total_steps}")
    logger.debug(f"Number of Epochs: {num_epochs}")
    
    # Setup model and callbacks.
    nerf, callbacks = setup_model_and_callbacks(params)

    nerf.fit(
        x = train_dataset, epochs = num_epochs,
        validation_data = val_dataset,
        validation_freq = params.system.validation_freq,
        callbacks = callbacks,
        steps_per_epoch = steps_per_epoch,
    )
    import pdb; pdb.set_trace()  # breakpoint 3ccdf13b //
