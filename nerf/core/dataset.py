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

        Args:
            imgs        :   A NumPy array of shape (N, H, W, 3)
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsic   :   A NumPy array of shape (3, 3)

        Returns:
            rays_o
            rays_d
            near
            far
            rgb
        """
        return rays_o, rays_d, near, far, rgb

    def create_tf_dataset(self, rays_o, rays_d, near, far, rgb):
        """
        Method that can be used by the subclasses.

        Returns a tf.data.Dataset object.

        Args:
            rays_o
            rays_d
            near
            far
            rgb

        Returns: 
            dataset

        TODO: Elaborate
        """
        return dataset

def CustomDataset(Dataset):
    """
    Custom Dataset
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
    
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
