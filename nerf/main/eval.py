import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from nerf.core import ops
from nerf.core.model import setup_model
from nerf.core.datasets import get_data

from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(logger, split):
    """
    TODO: Docstring
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)

    # Getting data
    data_splits, num_imgs = get_data(params)
    data = data_splits[split]
    
    H, W = data.imgs.shape[:2]
    zfill = int(np.log10(len(data.imgs)) + 5)
    psnr_vals = []

    # Rendering one image at a time and evaluating it.
    for i in tqdm(range(len(data.imgs)), desc = "Evaluating"):
        
        dataset = loader.create_dataset_for_render(
            H = H, W = W, c2w = data.poses[i], 
            bounds = data.bounds[i], intrinsic = data.intrinsic[i]
        )
        output = nerf.predict(x = dataset)

        fine_model_output = output[1]
        pred_rgb = fine_model_output["pred_rgb"]

        pred_rgb_numpy = pred_rgb.numpy()
        pred_rgb_numpy = np.clip(pred_rgb_numpy * 255.0, 0.0, 255.0)
        pred_img = pred_rgb_numpy.reshape(H, W, 3)

        psnr = psnr_metric_numpy(
            data.imgs[i].astype(np.float32), 
            pred_img.astype(np.float32)
        )
        psnr_vals.append(psnr)

        pred_img = cv2.cvtColor(pred_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        filename = {str(i).zfill(zfill)} + ".png"
        cv2.imwrite(os.path.join(render_params.save_dir, filename), pred_img)

    mean_psnr = np.mean(psnr)
    logger.info(f"Mean PSNR: {mean_psnr}")

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

    launch(logger, split = "test")
