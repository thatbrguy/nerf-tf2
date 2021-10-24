import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from nerf.core import ops
from nerf.core.model import setup_model
from nerf.core.datasets import setup_datasets

from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(logger, split):
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

    total_pixels = (img_HW[0] * img_HW[1] * num_imgs[split])
    total_steps = int(np.ceil(total_pixels / params.data.batch_size))

    logger.debug(f"Number of Steps: {total_steps}")
    
    # Setup model and callbacks.
    nerf = setup_model(params, num_imgs, img_HW)
    output = nerf.predict(x = tf_datasets[split], verbose = 1)

    fine_model_output = output[1]
    pred_rgb = fine_model_output["pred_rgb"]

    gt_rgb_ = [x[1][0] for x in tf_datasets[split]]
    gt_rgb = tf.concat(gt_rgb_, axis = 0)

    psnr = ops.psnr_metric(gt_rgb, pred_rgb)
    logger.info(f"PSNR: {psnr}")

    ## Saving Predictions.
    if not os.path.exists(params.eval.save_dir):
        os.makedirs(params.eval.save_dir, exist_ok=True)

    pred_imgs = tf.reshape(pred_rgb, [-1, img_HW[0], img_HW[1], 3])
    npy_pred_imgs = pred_imgs.numpy()
    npy_pred_imgs = np.clip(npy_pred_imgs * 255, 0, 255).astype(np.uint8)

    zfill = int(np.log10(num_imgs[split]) + 5)
    
    for i in tqdm(range(num_imgs[split]), desc = "Saving Images"):
        img = cv2.cvtColor(npy_pred_imgs[i], cv2.COLOR_RGB2BGR)
        path = os.path.join(params.eval.save_dir, str(i).zfill(zfill) + ".png")
        cv2.imwrite(path, img)

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
