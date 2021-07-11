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
    dataset = loader.get_mock_dataset()

    nerf = NeRF(params = params)
    # nerf_lite = NeRFLite(params)

    nerf.compile(optimizer = 'adam', metrics = [PSNRMetric()])
    nerf.fit(dataset, epochs = 1)
