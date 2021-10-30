import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from nerf.utils import pose_utils
from nerf.core.model import setup_model
from nerf.core.datasets import get_dataset_obj, CustomDataset
from nerf.utils.params_utils import load_params

os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

def launch(logger):
    """
    Launches the rendering run.
    """
    # Setting up params
    path = "./nerf/params/config.yaml"
    params = load_params(path)

    render_params = params.render
    assert render_params.num_cameras > 0

    if params.system.tf_seed is not None:
        tf.random.set_seed(params.system.tf_seed)

    if not os.path.exists(render_params.save_dir):
        os.makedirs(render_params.save_dir, exist_ok=True)
    
    dataset_obj = get_dataset_obj(params = params)
    nerf = setup_model(params)
    
    # Please refer to the documentation for more information 
    # about how the poses for rendering is calculated.
    poses = pose_utils.create_spherical_path(
        radius = render_params.radius, 
        num_cameras = render_params.num_cameras,
        inclination = render_params.inclination,
    )
    
    intrinsic = CustomDataset.camera_model_params_to_intrinsics(
        camera_model = render_params.camera_model_name,
        model_params = render_params.camera_model_params,
    )
    
    # If bounds is not specified in the config file, then default 
    # bounds are assumed. The default near bound is 25% of the 
    # diameter of the sphere. The default far bound is 75% of 
    # the diameter of the sphere.
    if render_params.bounds is None:
        diameter = 2 * render_params.radius 
        bounds = np.array([0.25 * diameter, 0.75 * diameter], dtype=np.float64)
    else:
        bounds = np.array(render_params.bounds, dtype=np.float32)

    H, W = render_params.img_size
    zfill = int(np.log10(render_params.num_cameras) + 5)

    # Rendering one image at a time.
    for i in tqdm(range(render_params.num_cameras), desc = "Rendering Images"):
        
        dataset = dataset_obj.create_dataset_for_render(
            H = H, W = W, c2w = poses[i], 
            bounds = bounds, intrinsic = intrinsic
        )
        output = nerf.predict(x = dataset)

        fine_model_output = output[1]
        pred_rgb = fine_model_output["pred_rgb"]

        pred_rgb = np.clip(pred_rgb * 255.0, 0.0, 255.0)
        pred_img = pred_rgb.reshape(H, W, 3).astype(np.uint8)

        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        filename = f"render_{str(i).zfill(zfill)}.png"
        cv2.imwrite(os.path.join(render_params.save_dir, filename), pred_img)

if __name__ == '__main__':

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
