import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
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

    def prepare_data(self, imgs, poses, bounds, intrinsic):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        
        TODO: Mention that it is also assumed that all images 
        have the same H, W.

        TODO: Do we normalize rgb range?

        N --> Number of images in the dataset.

        Args:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsic   :   A NumPy array of shape (3, 3)

        Returns:
            rays_o      :   A NumPy array of shape (N * H * W, 3)
            rays_d      :   A NumPy array of shape (N * H * W, 3)    
            near        :   A NumPy array of shape (N * H * W, 1) 
            far         :   A NumPy array of shape (N * H * W, 1)
            rgb         :   A NumPy array of shape (N * H * W, 3)

        """
        rays_o, rays_d, rgb = [], [], []
        K = intrinsic

        # Code currently only supports intrinsic matrices that are of the form:
        # intrinsic = np.array([
        #     [f, 0, c],
        #     [0, f, c],
        #     [0, 0, 1],
        # ])
        # In the above, f is the focal length and c is the principal offset. 
        ## TODO: Support intrinsic matrices which have distinct fu, fv, cu, cv.

        zero_check = np.array([
            K[0, 1], K[1, 0], K[2, 0], K[2, 1],
        ])
        assert (K[0, 0] == K[1, 1]) and (K[0, 2] == K[1, 2])
        assert K[2, 2] == 1
        assert np.all(zero_check == 0)

        # Code currently only works for H == W. TODO: Support H != W.
        assert self.params.H == self.params.W

        for idx, img in enumerate(imgs):
            rays_o_, rays_d_ = ray_utils.get_rays(
                H = self.params.H, W = self.params.W, 
                intrinsic = K, c2w = poses[idx],
            )
            rgb_ = img.reshape(-1, 3)

            rgb.append(rgb_)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)

        rgb = np.array(rgb)
        rays_o = np.array(rays_o)
        rays_d = np.array(rays_d)

        # (N, 2) --> (N, H*W, 2). TODO: Elaborate.
        bounds_ = np.broadcast_to(
            bounds[:, None, :], shape = (*rays_d.shape[:-1], 2)
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
        # Here, x has the input data, y had the target data.
        x = (rays_o, rays_d, near, far)
        y = (rgb,)        
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        ## TODO: Think about what dataset operations to add here.

        return dataset

class CustomDataset(Dataset):
    """
    Custom Dataset
    """
    ## Defining some camera model related class variables which will 
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

    @classmethod
    def camera_model_params_to_intrinsics(cls, cam_model, model_params):
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
        assert cam_model in cls.SUPPORTED_CAMERA_MODELS, \
            f"Camera model {cam_model} is not supported."

        if cam_model in cls.SAME_FOCAL:
            f, cx, cy = model_params[:3]
            fx, fy = f, f

        elif cam_model in cls.DIFF_FOCAL:
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

        Loads the images, poses, bounds, 
        intrinsics to memory.
        
        poses are camera to world transform (RT) matrices.

        N --> Number of images in the dataset. Column 0 in 
        bounds is near, column 1 in bounds is far.

        Same intrinsic matrix is assumed for all images.

        Returns:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsic   :   A NumPy array of shape (3, 3)
        """

        ########################################################
        ## Using mock data for testing!
        N, H, W = 10, 500, 500
        focal = 250

        intrinsics = np.eye(3)
        intrinsics[0, 0], intrinsics[1, 1] = focal, focal

        imgs = np.ones((N, H, W, 3))
        poses = np.ones((N, 4, 4))
        bounds = np.ones((N, 2))
        ########################################################

        return imgs, poses, bounds, intrinsics

    def _load_mock_data(self):
        """
        Loads mock data which can be used for testing functionality.

        Returns:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsic   :   A NumPy array of shape (3, 3)
        """
        N, H, W = 10, 500, 500
        focal = 250

        intrinsics = np.eye(3)
        intrinsics[0, 0], intrinsics[1, 1] = focal, focal

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

if __name__ ==  "__main__":

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    dataset = loader.get_dataset()
