import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from nerf.core.model import PSNRMetric, NeRF, NeRFLite
from nerf.core.dataset import CustomDataset
from nerf.utils.params_utils import load_params

if __name__ ==  "__main__":

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    dataset = loader.get_dataset()
    tf.random.set_seed(11)
    
    nerf = NeRF(params = params)
    # nerf_lite = NeRFLite(params)

    ## Enabling eager mode for ease of debugging. 
    ## NOTE: The current mock data is not suitable for debugging. 
    ## Have to use real data to debug. This code is only kept as 
    ## a placeholder until the real data is ready to use.
    nerf.compile(optimizer = 'adam', metrics = [PSNRMetric()], run_eagerly = True)
    nerf.fit(dataset, epochs = 1)
