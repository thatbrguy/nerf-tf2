import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC, abstractmethod
from nerf.utils import ray_utils, pose_utils

class Dataset(ABC):
    """
    A class that has methods that are common for all datasets. 
    This class must not be used directly!

    TODO: Elaborate.
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def get_dataset(self):
        """
        This needs to be implemented by every subclass.

        TODO: Elaborate
        """
        pass

    def _validate_intrinsic_matrix(self, K):
        """
        Code currently only supports intrinsic matrices that are of the form:
        intrinsic = np.array([
            [f, 0, c],
            [0, f, c],
            [0, 0, 1],
        ])
        In the above, f is the focal length and c is the principal offset. 
        
        TODO: Support intrinsic matrices which have distinct fu, fv, cu, cv.
        """
        zero_check = np.array([
            K[0, 1], K[1, 0], K[2, 0], K[2, 1],
        ])
        assert (K[0, 0] == K[1, 1]) and (K[0, 2] == K[1, 2])
        assert K[2, 2] == 1
        assert np.all(zero_check == 0)

    def prepare_data(self, imgs, poses, bounds, intrinsics):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        
        IMPORTANT: The current codebase requires all images MUST 
        have the same height and all images MUST have the same width.

        TODO: Do we normalize rgb range?

        N --> Number of images in the dataset.

        Args:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsics  :   A NumPy array of shape (N, 3, 3)

        Returns:
            rays_o      :   A NumPy array of shape (N * H * W, 3)
            rays_d      :   A NumPy array of shape (N * H * W, 3)    
            near        :   A NumPy array of shape (N * H * W, 1) 
            far         :   A NumPy array of shape (N * H * W, 1)
            rgb         :   A NumPy array of shape (N * H * W, 3)

        """
        rays_o, rays_d, rgb = [], [], []
        height_check, width_check = None, None

        new_poses, new_bounds = pose_utils.reconfigure_poses_and_bounds(
            old_poses = poses, 
            old_bounds = bounds,
            origin_method = self.params.preprocessing.origin_method,
        )

        for idx, img in enumerate(imgs):
            H, W = img.shape[:2]
            K = intrinsics[idx]

            ## TODO: Allow images to have different heights and 
            ## widths in the future. 
            if height_check is None:
                height_check = H
            else:
                assert height_check == H, (
                    "The current codebase requires all images "
                    "to have the same height."
                )

            if width_check is None:
                width_check = H
            else:
                assert width_check == W, (
                    "The current codebase requires all images "
                    "to have the same width."
                )

            ## TODO: Support H != W.
            assert H == W, "The current codebase requries H == W"
            self._validate_intrinsic_matrix(K = K)
            
            rays_o_, rays_d_ = ray_utils.get_rays(
                H = H, W = W, intrinsic = K, c2w = new_poses[idx],
            )

            rgb_ = img.reshape(-1, 3).astype(np.float32)
            
            # Diving by 255 so that the range of the rgb 
            # data is between [0, 1]
            rgb_ = rgb_ / 255

            rgb.append(rgb_)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)

        rgb = np.array(rgb)
        rays_o = np.array(rays_o)
        rays_d = np.array(rays_d)

        # (N, 2) --> (N, H*W, 2). TODO: Elaborate.
        bounds_ = np.broadcast_to(
            new_bounds[:, None, :], shape = (*rays_d.shape[:-1], 2)
        )
        near, far = bounds_[..., 0:1], bounds_[..., 1:2]

        # After reshaping, rays_o, rays_d and rgb will 
        # have shape (N * H * W, 3)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        
        # After reshaping, near and far will have shape (N * H * W, 1)
        near = near.reshape(-1, 1)
        far = far.reshape(-1, 1)

        return rays_o, rays_d, near, far, rgb

    def create_tf_dataset(self, rays_o, rays_d, near, far, rgb):
        """
        Method that can be used by the subclasses.

        Returns a tf.data.Dataset object.

        Args:
            rays_o      :   A NumPy array of shape (N * H * W, 3)
            rays_d      :   A NumPy array of shape (N * H * W, 3)    
            near        :   A NumPy array of shape (N * H * W, 1) 
            far         :   A NumPy array of shape (N * H * W, 1)
            rgb         :   A NumPy array of shape (N * H * W, 3)

        Returns: 
            dataset

        TODO: Elaborate
        """
        ## NOTE: Maybe add a comment on memory?
        if self.params.data.pre_shuffle:

            rng = np.random.default_rng(
                seed = self.params.data.pre_shuffle_seed
            )
            perm = rng.permutation(len(rays_o))

            rays_o = rays_o[perm]
            rays_d = rays_d[perm]
            near = near[perm]
            far = far[perm]
            rgb = rgb[perm]

        ## TODO: A more elegant way if possible?
        rays_o = rays_o.astype(np.float32)
        rays_d = rays_d.astype(np.float32)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb.astype(np.float32)

        # Here, x has the input data, y had the target data.
        x = (rays_o, rays_d, near, far)
        y = (rgb,)        
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = True,
        )

        ## TODO: Think about what dataset operations to add here.
        return dataset

class CustomDataset(Dataset):
    """
    Custom Dataset
    """
    ## Defining some camera model related class attributes which will 
    ## be useful for some functions.

    # Camera models are as defined in
    # https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
    SUPPORTED_CAMERA_MODELS = frozenset([
        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", 
        "RADIAL", "OPENCV", "FULL_OPENCV",
    ])
    
    # Camera models which only have one focal length (f) in their 
    # model parameters (in this case the codebase will assume fx == fy == f).
    SAME_FOCAL = frozenset(["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"])
    
    # Camera models which provide both fx and fy.
    DIFF_FOCAL = frozenset(["PINHOLE", "OPENCV", "FULL_OPENCV"])

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.dataset_params = self.params.data.custom_dataset

    @classmethod
    def camera_model_params_to_intrinsics(cls, camera_model, model_params):
        """
        Given camera model name and camera model params, this function 
        creates and returns an intrinsic matrix.

        Camera models are used as defined in
        https://github.com/colmap/colmap/blob/master/src/base/camera_models.h

        The distortion paramters are not used since this function is only 
        interested in calculating the intrinsic matrix. Only the focal length 
        values and principal offset values are needed to create the intrinsic
        matrix.

        The created intrinsic matrix is of the form:

        intrinsic = np.array([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.],
        ], dtype = np.float64)

        TODO: Args and Returns.
        """
        assert camera_model in cls.SUPPORTED_CAMERA_MODELS, \
            f"Camera model {camera_model} is not supported."

        if camera_model in cls.SAME_FOCAL:
            f, cx, cy = model_params[:3]
            fx, fy = f, f

        elif camera_model in cls.DIFF_FOCAL:
            fx, fy, cx, cy = model_params[:4]

        else:
            raise RuntimeError(
                "This should not happen. Please check if "
                "SUPPORTED_CAMERA_MODELS, SAME_FOCAL and DIFF_FOCAL "
                "class attributes of the class CustomDataset are "
                "configured correctly."
            )

        intrinsic = np.array([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.],
        ], dtype = np.float64)

        return intrinsic
    
    def _load_full_dataset(self):
        """
        TODO: Elaborate.

        Loads the images, poses, bounds, intrinsics to memory.
        
        poses are camera to world transform (RT) matrices.

        N --> Number of images in the dataset. Column 0 in 
        bounds is near, column 1 in bounds is far.

        Same intrinsic matrix is assumed for all images.

        Returns:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsics  :   A NumPy array of shape (N, 3, 3)
        """
        imgs, poses, bounds, intrinsics = [], [], [], []

        csv_path = self.dataset_params.pose_info_path
        img_root = self.dataset_params.img_root_dir
        df = pd.read_csv(csv_path)

        for idx, row in enumerate(df.itertuples()):
            filename = row.image_name
            path = os.path.join(img_root, filename)

            img = cv2.imread(path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            camera_model = row.camera_model
            camera_params = np.array(yaml.safe_load(row.camera_params))
            intrinsic = self.camera_model_params_to_intrinsics(
                camera_model = camera_model, 
                model_params = camera_params
            )

            values = np.array(yaml.safe_load(row.pose)).reshape(3, 4)
            RT = pose_utils.make_4x4(values)

            bound = [row.near, row.far]

            imgs.append(img)
            poses.append(RT)
            bounds.append(bound)
            intrinsics.append(intrinsic)

        imgs = np.array(imgs)
        poses = np.array(poses)
        bounds = np.array(bounds)
        intrinsics = np.array(intrinsics)

        return imgs, poses, bounds, intrinsics

    def _load_mock_dataset(self):
        """
        Loads mock data which can be used for testing functionality.

        NOTE: The mock data may not be suitable for testing the 
        working of the algorithms. I mostly used it to test if the 
        codebase throws errors. Please use this function with extreme 
        caution. Please use real data for proper testing.

        THIS FUNCTION WILL BE REMOVED SOON. PLEASE USE THE REAL 
        DATA ITSELF FOR TESTING.

        Returns:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsic   :   A NumPy array of shape (N, 3, 3)
        """
        N, H, W = 10, 500, 500
        focal = 250

        K = np.eye(3)
        K[0, 0], K[1, 1] = focal, focal
        K[0, 2], K[1, 2] = W/2, H/2
        intrinsics = np.array([K for _ in range(N)])

        imgs = np.ones((N, H, W, 3))
        poses = np.ones((N, 4, 4))
        bounds = np.ones((N, 2))

        return imgs, poses, bounds, intrinsics

    def get_dataset(self):
        """
        TODO: Elaborate.
        """
        (
            imgs, poses, 
            bounds, intrinsic
        ) = self._load_full_dataset()

        (
            rays_o, rays_d, 
            near, far, rgb
        ) = super().prepare_data(imgs, poses, bounds, intrinsic)

        dataset = super().create_tf_dataset(
            rays_o, rays_d, near, far, rgb
        )

        return dataset

    def get_mock_dataset(self):
        """
        Loads mock data which can be used for testing functionality.

        NOTE: The mock data may not be suitable for testing the 
        working of the algorithms. I mostly used it to test if the 
        codebase throws errors. Please use this function with extreme 
        caution. Please use real data for proper testing.

        THIS FUNCTION WILL BE REMOVED SOON. PLEASE USE THE REAL 
        DATA ITSELF FOR TESTING.
        """
        (
            imgs, poses, 
            bounds, intrinsic
        ) = self._load_mock_dataset()

        (
            rays_o, rays_d, 
            near, far, rgb
        ) = super().prepare_data(imgs, poses, bounds, intrinsic)

        dataset = super().create_tf_dataset(
            rays_o, rays_d, near, far, rgb
        )

        return dataset

if __name__ ==  "__main__":

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    dataset = loader.get_dataset()
    import pdb; pdb.set_trace()  # breakpoint c051f513 //
