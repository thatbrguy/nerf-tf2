import logging

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import namedtuple
from nerf.utils import ray_utils, pose_utils

# Setting up logger.
logger = logging.getLogger(__name__)

# Setting up DataContainer.
DataContainer = namedtuple(
    'DataContainer', ("imgs", "poses", "bounds", "intrinsics")
)

class Dataset(ABC):
    """
    A class that has methods that are common for all datasets. 
    This class must not be used directly!

    TODO: Elaborate.
    """
    def __init__(self, params):
        self.params = params
        self.splits = ["train", "test", "val"]

    @abstractmethod
    def get_dataset(self):
        """
        This needs to be implemented by every subclass.

        TODO: Elaborate
        """
        pass

    @staticmethod
    def _validate_intrinsic_matrix(K):
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

    def _validate_all_splits(self, data_splits):
        """
        TODO: Docstring
        """
        all_H, all_W = [], []
        for split in self.splits:
            data = data_splits[split]
            img, intrinsics = data.img, data.intrinsics

            for idx in range(len(intrinsics)):
                K = intrinsics[idx]
                img = imgs[idx]

                H, W = img.shape[:2]
                all_H.append(H)
                all_W.append(W)
                self._validate_intrinsic_matrix(K = K)

        all_H, all_W = np.array(all_H), np.array(all_W)
        assert np.all(all_H == all_H[0])
        assert np.all(all_W == all_W[0])

    def _reconfigure_imgs_and_intrinsics(self, data_splits):
        """
        TODO: Docstring
        """
        reconf_data_splits = dict()
        
        for split in self.splits:
            data = data_splits[split]
            
            new_imgs, new_intrinsics = pose_utils.scale_imgs_and_intrinsics(
                old_imgs = data.imgs, old_intrinsics = data.intrinsics, 
                scale_factor = self.params.data.scale_imgs,
            )

            reconf_data = DatasetContainer(
                imgs = new_imgs, poses = data.poses, 
                bounds = data.bounds, intrinsics = new_intrinsics,
            )
            reconf_data_splits[split] = reconf_data

        return reconf_data_splits

    def _reconfigure_poses(self, data_splits):
        """
        TODO: Docstring
        """
        reconf_data_splits = dict()
        
        all_poses = [data[x].poses for x in self.splits]
        all_poses = np.concatenate(all_poses, axis = 0)

        W2_to_W1_transform = pose_utils.calculate_new_world_pose(
            poses = all_poses, 
            origin_method = self.params.preprocessing.origin_method,
        )

        for split in self.splits:
            data = data_splits[split]

            new_poses = pose_utils.reconfigure_poses(
                old_poses = data.poses, 
                W2_to_W1_transform = W2_to_W1_transform
            )
            new_data = DatasetContainer(
                imgs = data.imgs, poses = new_poses, 
                bounds = data.bounds, intrinsics = data.intrinsics,
            )
            reconf_data_splits[split] = new_data

        return reconf_data_splits

    def _reconfigure_scene_scale(self, data_splits):
        """
        TODO: Docstring
        """
        reconf_data_splits = dict()
        
        # Currently, we make sure that all images in all splits have the same 
        # height (H) and width (W). Hence, we can take the H, W of an arbitrary 
        # image for our purposes.
        H, W = data_splits["train"].imgs[0].shape[:2]
        all_poses = [data[x].poses for x in self.splits]
        all_bounds = [data[x].bounds for x in self.splits]
        all_intrinsics = [data[x].intrinsics for x in self.splits]

        all_poses = np.concatenate(all_poses, axis = 0)
        all_bounds = np.concatenate(all_bounds, axis = 0)
        all_intrinsics = np.concatenate(all_intrinsics, axis = 0)

        ## TODO: Rename function name
        scene_scale_factor = pose_utils.calculate_scene_scale(
            poses = all_poses, bounds = all_bounds, 
            origin_method = self.params.preprocessing.bounds_method,
            intrinsics = all_intrinsics, height = H, width = W
        )

        for split in self.splits:
            data = data_splits[split]

            new_poses, new_bounds = pose_utils.reconfigure_scene_scale(
                old_poses = data.poses, old_bounds = data.bounds, 
                scene_scale_factor = scene_scale_factor,
            )
            new_data = DatasetContainer(
                imgs = data.imgs, poses = new_poses, 
                bounds = new_bounds, intrinsics = data.intrinsics,
            )
            reconf_data_splits[split] = new_data

        return reconf_data_splits

    def validate_and_reconfigure_data(self, data_splits):
        """
        TODO: Docstring.
        """
        self._validate_all_splits(data_splits)

        temp_data_step_1 = self._reconfigure_imgs_and_intrinsics(data_splits)
        temp_data_step_2 = self._reconfigure_poses(temp_data_step_1)
        final_data_splits = self._reconfigure_scene_scale(temp_data_step_2)

        return final_data_splits

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
        data = self.validate_and_reconfigure_data(
            imgs = imgs, poses = poses, 
            bounds = bounds, intrinsics = intrinsics,
        )

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
        train_dataset = train_dataset.repeat(
            count = self.params.data.repeat_count
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        logger.debug("Created TensorFlow datasets.")

        ## TODO: Think about what dataset operations to add here.
        return train_dataset, val_dataset, train_spec, val_spec

