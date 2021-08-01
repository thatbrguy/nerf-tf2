import os
import cv2
import time
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

def setup_model(params):
    """
    Sets up the NeRF model.
    """
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

    return nerf

if __name__ == "__main__":

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

    loader = CustomDataset(params = params)
    nerf = setup_model(params)
    
    imgs, poses, bounds, intrinsics = loader.get_reconfigured_data()

    # data = (intrinsic, c2w, H, W)
    data = (
        intrinsics[0].astype(np.float32), poses[0].astype(np.float32), 
        imgs[0].shape[0], imgs[0].shape[1]
    )

    logger.debug("Starting to render.")
    start = time.time()
    
    img = nerf.render(data = data)
    end = time.time()
    logger.debug(f"Render complete. Took {(end-start):.3f}s.")
    
    cv2.imwrite("./render.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
