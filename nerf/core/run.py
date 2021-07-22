import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from nerf.core import ops
from nerf.core.model import NeRF, NeRFLite
from nerf.core.dataset import CustomDataset

from nerf.utils.params_utils import load_params

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

    ## TODO: Monitor the PSNR! Also, consider having filepath with formatting.
    model_ckpt = ModelCheckpoint(
        filepath = params.model.save_path, 
        monitor = "val_psnr", save_best_only = True
    )
    psnr = ops.PSNRMetric()
    tensorboard = TensorBoard()

    ## TODO: Setup val_spec!
    val_imgs_logger = ops.LogValImages(params = params, val_spec = val_spec)

    # Enabling eager mode for ease of debugging. 
    nerf.compile(optimizer = 'adam', metrics = [psnr], run_eagerly = True)
    nerf.fit(
        x = train_dataset, epochs = 2,
        validation_data = val_dataset,
        validation_freq = 1,
        callbacks = [model_ckpt, tensorboard, val_imgs_logger],
        steps_per_epoch = 10,
    )
