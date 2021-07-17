import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from nerf.core.model import PSNRMetric, NeRF, NeRFLite, psnr_metric
from nerf.core.dataset import CustomDataset
from nerf.utils.params_utils import load_params

if __name__ ==  "__main__":

    np.set_printoptions(precision = 5, suppress = True)
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    train_dataset, val_dataset = loader.get_dataset()
    tf.random.set_seed(11)
    
    nerf = NeRF(params = params)
    # nerf_lite = NeRFLite(params)

    ## TODO: Monitor the PSNR! Also, consider having filepath with formatting.
    model_ckpt = ModelCheckpoint(
        filepath = params.model.save_path, 
        monitor = "val_loss", save_best_only = True
    )
    tensorboard = TensorBoard()

    # Enabling eager mode for ease of debugging. 
    nerf.compile(optimizer = 'adam', metrics = [PSNRMetric()], run_eagerly = True)
    nerf.fit(
        x = train_dataset, epochs = 1,
        validation_data = val_dataset,
        callbacks = [model_ckpt, tensorboard],
    )
