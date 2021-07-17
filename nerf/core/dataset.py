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
        Code supports intrinsic matrices that are of the form:
        intrinsic = np.array([
            [fu, 0., cu],
            [0., fv, cv],
            [0., 0., 1.],
        ])
        """
        zero_check = np.array([
            K[0, 1], K[1, 0], K[2, 0], K[2, 1],
        ])
        assert K[2, 2] == 1
        assert np.all(zero_check == 0)

    def _split(self, x, y, frac):
        """
        Splits the data into two splits.

        Args:
            x       :   A tuple with NumPy arrays. The tuple 
                        should contain the following NumPy 
                        arrays:
                            x = (rays_o, rays_d, near, far)

            y       :   A tuple with NumPy arrays. The tuple 
                        should contain the following NumPy 
                        array:
                            y = (rgb,)
            
            frac    :   A float value in the range (0, 1) which 
                        indicates the fraction of the data that 
                        should belong in the second split. 

        Returns:
            x_split_1   :   TODO: Elaborate
            x_split_2   :   TODO: Elaborate
            y_split_1   :   TODO: Elaborate
            y_split_2   :   TODO: Elaborate
        """
        (rays_o, rays_d, near, far) = x
        (rgb,) = y

        size = len(rays_o)

        # Legend --> s1: Split 1, s2: Split 2
        s2_size = int(size * frac)
        s1_size = size - split_2_size

        s1_rays_o, s2_rays_o = rays_o[:s1_size], rays_o[s1_size:]
        s1_rays_d, s2_rays_d = rays_d[:s1_size], rays_d[s1_size:]
        s1_near, s2_near = near[:s1_size], near[s1_size:]
        s1_far, s2_far = far[:s1_size], far[s1_size:]
        s1_rgb, s2_rgb = rgb[:s1_size], rgb[s1_size:]

        x_split_1 = (s1_rays_o, s1_rays_d, s1_near, s1_far)
        x_split_2 = (s2_rays_o, s2_rays_d, s2_near, s2_far)
        y_split_1 = (s1_rgb,)
        y_split_2 = (s2_rgb,)

        return x_split_1, x_split_2, y_split_1, y_split_2

    def prepare_data(self, imgs, poses, bounds, intrinsics):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        
        Each image can have a different height and width value. 
        If image i has width W_i and height H_i, then the number 
        of pixels in that image is Hi*Wi. Hence, total number 
        of pixels in the dataset (L) is given by:

        L = H_0*W_0 +  H_1*W_1 +  H_2*W_2 + ... + H_(N-1)*W_(N-1)

        Legend:
            N: Number of images in the dataset.
            L: Total number of pixels in the dataset. 

        Args:
            imgs        :   A list of N NumPy arrays. Each element
                            of the list is a NumPy array with shape
                            (H_i, W_i, 3) that represents an image.
                            Each image can have different height
                            and width.
            poses       :   A NumPy array of shape (N, 4, 4)
            bounds      :   A NumPy array of shape (N, 2)
            intrinsics  :   A NumPy array of shape (N, 3, 3)

        Returns:
            rays_o      :   A NumPy array of shape (L, 3)
            rays_d      :   A NumPy array of shape (L, 3)    
            near        :   A NumPy array of shape (L, 1) 
            far         :   A NumPy array of shape (L, 1)
            rgb         :   A NumPy array of shape (L, 3)

        """
        rays_o, rays_d, near, far, rgb = [], [], [], [], []

        new_poses, new_bounds = pose_utils.reconfigure_poses_and_bounds(
            old_poses = poses, 
            old_bounds = bounds,
            origin_method = self.params.preprocessing.origin_method,
        )

        new_imgs, new_intrinsics = pose_utils.scale_imgs_and_intrinsics(
            old_imgs = imgs, old_intrinsics = intrinsics, 
            scale_factor = self.params.data.scale_imgs,
        )

        for idx in range(len(new_imgs)):
            
            img = new_imgs[idx]
            K = new_intrinsics[idx]
            self._validate_intrinsic_matrix(K = K)

            H, W = img.shape[:2]
            rays_o_, rays_d_ = ray_utils.get_rays(
                H = H, W = W, intrinsic = K, c2w = new_poses[idx],
            )
            
            # Diving by 255 so that the range of the rgb 
            # data is between [0, 1]
            rgb_ = img.reshape(-1, 3).astype(np.float32)
            rgb_ = rgb_ / 255

            rgb.append(rgb_)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)

            bounds_ = new_bounds[idx]
            bounds_ = np.broadcast_to(
                bounds_[None, :], shape = (rays_d_.shape[0], 2)
            )
            near_, far_ = bounds_[:, 0:1], bounds_[:, 1:2]

            near.append(near_)
            far.append(far_)

        rgb = np.concatenate(rgb, axis = 0)
        far = np.concatenate(far, axis = 0)
        near = np.concatenate(near, axis = 0)
        rays_o = np.concatenate(rays_o, axis = 0)
        rays_d = np.concatenate(rays_d, axis = 0)

        return rays_o, rays_d, near, far, rgb

    def create_tf_dataset(self, rays_o, rays_d, near, far, rgb):
        """
        Method that can be used by the subclasses.

        Returns a tf.data.Dataset object.
        
        Each image can have a different height and width value. 
        If image i has width W_i and height H_i, then the number 
        of pixels in that image is Hi*Wi. Hence, total number 
        of pixels in the dataset (L) is given by:

        L = H_0*W_0 +  H_1*W_1 +  H_2*W_2 + ... + H_(N-1)*W_(N-1)

        Legend:
            L: Total number of pixels in the dataset.

        Args:
            rays_o      :   A NumPy array of shape (L, 3)
            rays_d      :   A NumPy array of shape (L, 3)    
            near        :   A NumPy array of shape (L, 1) 
            far         :   A NumPy array of shape (L, 1)
            rgb         :   A NumPy array of shape (L, 3)

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

        # Here, x has the input data, y has the target data.
        x = (rays_o, rays_d, near, far)
        y = (rgb,)        
        
        (
            x_train, x_val, 
            y_train, y_val,
        ) = self._split(x = x, y = y, frac = self.params.data.val_split)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = True,
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = True,
        )

        ## TODO: Think about what dataset operations to add here.
        return train_dataset, val_dataset

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

        Returns:
            imgs        :   A list of N NumPy arrays. Each element
                            of the list is a NumPy array with shape
                            (H_i, W_i, 3) that represents an image.
                            Each image can have different height
                            and width.
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

        poses = np.array(poses)
        bounds = np.array(bounds)
        intrinsics = np.array(intrinsics)

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

        train_dataset, val_dataset = super().create_tf_dataset(
            rays_o, rays_d, near, far, rgb
        )

        return train_dataset, val_dataset

if __name__ ==  "__main__":

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    dataset = loader.get_dataset()
    import pdb; pdb.set_trace()  # breakpoint c051f513 //
