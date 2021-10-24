import os
import cv2
from tqdm import tqdm
import tensorflow as tf

from nerf.core.datasets import get_dataset_obj, CustomDataset
from nerf.core.utils import pose_utils
from nerf.core.model import setup_model

if __name__ == '__main__':

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)
    render_params = params.system.render
    assert render_params.num_cameras > 0

    if not os.path.exists(render_params.save_dir):
        os.makedirs(render_params.save_dir, exist_ok=True)
    
    loader = get_dataset_obj(params = params)
    nerf = setup_model(params)
    
    poses = pose_utils.create_spherical_path(
        radius = render_params.radius, 
        num_cameras = render_params.num_cameras,
        inclination = render_params.inclination,
    )
    intrinsic = CustomDataset.camera_model_params_to_intrinsics(
        camera_model = render_params.camera_model_name,
        model_params = render_params.camera_model_params,
    )
    if render_params.bounds is None:
        diameter = 2 * render_params.radius 
        bounds = np.array([0.25 * diameter, 0.75 * diameter], dtype=np.float64)
    else:
        bounds = np.array(render_params.bounds, dtype=np.float64)

    H, W = render_params.img_size
    zfill = int(np.log10(render_params.num_cameras) + 5)

    # Rendering one image at a time.
    for i in tqdm(range(render_params.num_cameras), desc = "Rendering Images"):
        
        dataset = loader.create_dataset_for_render(
            H = H, W = W, c2w = poses[i], 
            bounds = bounds, intrinsic = intrinsic
        )
        output = nerf.predict(x = dataset)

        fine_model_output = output[1]
        pred_rgb = fine_model_output["pred_rgb"]

        pred_rgb_numpy = pred_rgb.numpy()
        pred_rgb_numpy = np.clip(pred_rgb_numpy * 255.0, 0.0, 255.0)
        pred_img = pred_rgb_numpy.reshape(H, W, 3).astype(np.uint8)

        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        filename = {str(i).zfill(zfill)} + ".png"
        cv2.imwrite(os.path.join(render_params.save_dir, filename), pred_img)
