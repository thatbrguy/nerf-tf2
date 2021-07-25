import os
import cv2
import yaml
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC, abstractmethod
from nerf.utils import ray_utils, pose_utils

# Setting up logger.
logger = logging.getLogger(__name__)

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

    def _split(self, data, frac = None, num = None):
        """
        Splits the data into two splits.

        Either frac or num MUST be provided. Only one of 
        them must be provided.

        Args:
            data    :   A tuple with NumPy arrays or lists. 
                        TODO: Mention len constraint.
            
            frac    :   A float value in the range (0, 1) which 
                        indicates the fraction of the data that 
                        should belong in the second split. If 
                        None, then num MUST be provided.

            num     :   A positive integer which indicates 
                        indicates the number of data elements 
                        that should belong in the second split. 
                        If None, then frac MUST be provided.

        Returns:
            split_1     :   TODO: Elaborate
            split_2     :   TODO: Elaborate
        """
        assert ((frac is None) ^ (num is None)), (
            "Please provide either frac or num. Only one of "
            "them must be provided."
        )
        shape_check = np.array([len(arr) for arr in data])
        assert np.all(shape_check == shape_check[0])

        # Legend --> s1: Split 1, s2: Split 2
        size = len(data[0])
        if frac is not None:
            assert (type(frac) is float) and (frac > 0) and (frac < 1)
            s2_size = int(size * frac)
        else:
            assert (type(num) is int) and (num > 0)
            s2_size = num

        s1_size = size - s2_size
        split_1, split_2 = [], []
        
        for arr in data:
            s1_arr, s2_arr = arr[:s1_size], arr[s1_size:]
            split_1.append(s1_arr)
            split_2.append(s2_arr)

        split_1 = tuple(split_1)
        split_2 = tuple(split_2)

        return split_1, split_2

    def process_data(self, imgs, poses, bounds, intrinsics):
        """
        Creates rays_o, rays_d, near, far, rgb

        TODO: Elaborate.
        
        Each image can have a different height and width value. 
        If image i has width W_i and height H_i, then the number 
        of pixels in that image is Hi*Wi. Hence, total number 
        of pixels (here, L) is given by:

        L = H_0*W_0 +  H_1*W_1 +  H_2*W_2 + ... + H_(N-1)*W_(N-1)

        Legend:
            N: Number of images.
            L: Total number of pixels. 

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
            A list containing the following:
            rays_o      :   A NumPy array of shape (L, 3)
            rays_d      :   A NumPy array of shape (L, 3)
            near        :   A NumPy array of shape (L, 1) 
            far         :   A NumPy array of shape (L, 1)
            rgb         :   A NumPy array of shape (L, 3)
            spec        :   A list of length N which contains 
                            the height and width information 
                            of each image.
        """
        rays_o, rays_d, near = [], [], []
        far, rgb, spec = [], [], []
        
        for idx in range(len(imgs)):
            
            img = imgs[idx]
            K = intrinsics[idx]

            H, W = img.shape[:2]
            rays_o_, rays_d_ = ray_utils.get_rays(
                H = H, W = W, intrinsic = K, c2w = poses[idx],
            )
            
            # Diving by 255 so that the range of the rgb 
            # data is between [0, 1]
            rgb_ = img.reshape(-1, 3).astype(np.float32)
            rgb_ = rgb_ / 255

            rgb.append(rgb_)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)

            bounds_ = bounds[idx]
            bounds_ = np.broadcast_to(
                bounds_[None, :], shape = (rays_d_.shape[0], 2)
            )
            near_, far_ = bounds_[:, 0:1], bounds_[:, 1:2]

            near.append(near_)
            far.append(far_)
            spec.append((H, W))

        rgb = np.concatenate(rgb, axis = 0)
        far = np.concatenate(far, axis = 0)
        near = np.concatenate(near, axis = 0)
        rays_o = np.concatenate(rays_o, axis = 0)
        rays_d = np.concatenate(rays_d, axis = 0)

        return [rays_o, rays_d, near, far, rgb, spec]

    def prepare_data(self, imgs, poses, bounds, intrinsics):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.

        Legend:
            N: Number of images.
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
            train_data  : TODO: Elaborate
            val_data    : TODO: Elaborate

        """
        # Validating intrinsic matrices.
        for idx in range(len(intrinsics)):
            K = intrinsics[idx]
            self._validate_intrinsic_matrix(K = K)

        new_poses, new_bounds = pose_utils.reconfigure_poses_and_bounds(
            old_poses = poses, 
            old_bounds = bounds,
            origin_method = self.params.preprocessing.origin_method,
        )

        new_imgs, new_intrinsics = pose_utils.scale_imgs_and_intrinsics(
            old_imgs = imgs, old_intrinsics = intrinsics, 
            scale_factor = self.params.data.scale_imgs,
        )

        data = (new_imgs, new_poses, new_bounds, new_intrinsics)
        train_data, val_data = self._split(
            data = data, 
            frac = self.params.data.validation.frac_imgs, 
            num = self.params.data.validation.num_imgs,
        )

        train_proc = self.process_data(*train_data)
        val_proc = self.process_data(*val_data)

        return train_proc, val_proc

    def create_tf_dataset(self, train_proc, val_proc):
        """
        Method that can be used by the subclasses.
        
        Each image can have a different height and width value. 
        If image i has width W_i and height H_i, then the number 
        of pixels in that image is Hi*Wi. Hence, total number 
        of pixels in the dataset (L) is given by:

        L = H_0*W_0 +  H_1*W_1 +  H_2*W_2 + ... + H_(N-1)*W_(N-1)

        Legend:
            L: Total number of pixels in the dataset.

        Args:
            TODO

        Returns: 
            TODO
        
        Shapes:
            rays_o      :   A NumPy array of shape (L, 3)
            rays_d      :   A NumPy array of shape (L, 3)    
            near        :   A NumPy array of shape (L, 1) 
            far         :   A NumPy array of shape (L, 1)
            rgb         :   A NumPy array of shape (L, 3)

        TODO: Elaborate
        """
        logger.debug("Creating TensorFlow datasets.")
        
        x_train, y_train = train_proc[:4], train_proc[4:5]
        x_val, y_val = val_proc[:4], val_proc[4:5]
        train_spec, val_spec = train_proc[5], val_proc[5]

        ## NOTE: Maybe add a comment on memory?
        if self.params.data.pre_shuffle:

            rng = np.random.default_rng(
                seed = self.params.data.pre_shuffle_seed
            )
            perm = rng.permutation(len(x_train[0]))

            # We only shuffle the train dataset. We do not 
            # shuffle the val dataset.
            x_train[0] = x_train[0][perm]
            x_train[1] = x_train[1][perm]
            x_train[2] = x_train[2][perm]
            x_train[3] = x_train[3][perm]
            y_train[0] = y_train[0][perm]

        ## TODO: A more elegant way if possible?
        x_train[0] = x_train[0].astype(np.float32)
        x_train[1] = x_train[1].astype(np.float32)
        x_train[2] = x_train[2].astype(np.float32)
        x_train[3] = x_train[3].astype(np.float32)
        y_train[0] = y_train[0].astype(np.float32)

        x_val[0] = x_val[0].astype(np.float32)
        x_val[1] = x_val[1].astype(np.float32)
        x_val[2] = x_val[2].astype(np.float32)
        x_val[3] = x_val[3].astype(np.float32)
        y_val[0] = y_val[0].astype(np.float32)

        x_train, y_train = tuple(x_train), tuple(y_train)
        x_val, y_val = tuple(x_val), tuple(y_val)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = True,
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        logger.debug("Created TensorFlow datasets.")

        ## TODO: Think about what dataset operations to add here.
        return train_dataset, val_dataset, train_spec, val_spec

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
        logger.debug("Loading custom dataset.")
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

            ## Using a smaller dataset for the time being. TODO: Remove this!
            # if idx == 10:
            #     break

        poses = np.array(poses)
        bounds = np.array(bounds)
        intrinsics = np.array(intrinsics)

        logger.debug("Loaded custom dataset.")

        return imgs, poses, bounds, intrinsics

    def get_dataset(self):
        """
        TODO: Elaborate.
        """
        (
            imgs, poses, 
            bounds, intrinsics
        ) = self._load_full_dataset()

        train_proc, val_proc = super().prepare_data(
            imgs, poses, bounds, intrinsics
        )

        (
            train_dataset, val_dataset, 
            train_spec, val_spec
        ) = super().create_tf_dataset(
            train_proc, val_proc
        )

        return train_dataset, val_dataset, train_spec, val_spec

if __name__ ==  "__main__":

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    loader = CustomDataset(params = params)
    (
        train_dataset, val_dataset, 
        train_spec, val_spec
    ) = loader.get_dataset()
    import pdb; pdb.set_trace()  # breakpoint bf3ff96d //
