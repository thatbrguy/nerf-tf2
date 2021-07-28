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

if __name__ ==  "__main__":

    # Setting numpy print options for ease of debugging.
    np.set_printoptions(precision = 5, suppress = True)

    # Setting up logger.
    logging.basicConfig(
        format='[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG
    )
    logger = logging.getLogger()

    # Setting TF seed to enable determinism of TF.
    tf.random.set_seed(11)
    
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    train_dataset, val_dataset, train_spec, val_spec = loader.get_dataset()
    
    nerf = NeRF(params = params)
    # nerf_lite = NeRFLite(params)

    # model_ckpt = ModelCheckpoint(
    #     filepath = params.model.save_path, 
    #     monitor = "val_psnr_metric", 
    #     save_best_only = True, mode = "max"
    # )

    model_ckpt = ops.CustomModelSaver(params = params, save_best_only = False)
    tensorboard = TensorBoard(update_freq = "epoch")

    ## NOTE: The val images logger is not working as 
    ## expected and hence is not currently being used. 
    ## Will think about an alternative way to log images later.
    # val_imgs_logger = ops.LogValImages(params = params, val_spec = val_spec)

    # Total number of pixels when the entire dataset is repeated 5 times:
    total_pixels = (800 * 800 * 100 * 5)
    # Total number of steps:
    total_steps = int(total_pixels / params.data.batch_size)
    # Steps per epoch:
    steps_per_epoch = 128
    # Number of epochs:
    num_epochs = int(total_steps / steps_per_epoch)

    logger.debug(f"Number of Steps: {total_steps}")
    logger.debug(f"Number of Epochs: {num_epochs}")

    ## Setting up the optimizer and LR scheduler.
    lr_schedule = ExponentialDecay(
        initial_learning_rate = 5e-4,
        decay_steps = 200000,
        decay_rate = 0.1,
    )
    optimizer = Adam(learning_rate = lr_schedule)

    # Enabling eager mode for ease of debugging. 
    nerf.compile(
        optimizer = optimizer, 
        metrics = [ops.psnr_metric], run_eagerly = True,
    )

    if params.model.load.set_weights:
        nerf.set_everything()
    
    nerf.fit(
        x = train_dataset, epochs = 3, # num_epochs,
        validation_data = val_dataset,
        validation_freq = 1, # 100
        callbacks = [model_ckpt, tensorboard],
        steps_per_epoch = 1, # steps_per_epoch,
    )
    import pdb; pdb.set_trace()  # breakpoint 3ccdf13b //
