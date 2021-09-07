import logging

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import namedtuple
from nerf.utils import ray_utils, pose_utils

# Setting up logger.
logger = logging.getLogger(__name__)

# Setting up ContainerType1.
ContainerType1 = namedtuple(
    'ContainerType1', ("imgs", "poses", "bounds", "intrinsics")
)

# Setting up ContainerType2.
ContainerType2 = namedtuple(
    'ContainerType2', ("rays_o", "rays_d", "near", "far", "rgb")
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

        self._scale_mul = self.params.data.scene_scale_mul
        self._scale_add = self.params.data.scene_scale_add

        self._scale_mul = 1.0 if self._scale_mul is None else self._scale_mul
        self._scale_add = 0.0 if self._scale_add is None else self._scale_add

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
            imgs, intrinsics = data.imgs, data.intrinsics

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

            reconf_data = ContainerType1(
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
        
        all_poses = [data_splits[x].poses for x in self.splits]
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
            new_data = ContainerType1(
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
        all_poses = [data_splits[x].poses for x in self.splits]
        all_bounds = [data_splits[x].bounds for x in self.splits]
        all_intrinsics = [data_splits[x].intrinsics for x in self.splits]

        all_poses = np.concatenate(all_poses, axis = 0)
        all_bounds = np.concatenate(all_bounds, axis = 0)
        all_intrinsics = np.concatenate(all_intrinsics, axis = 0)

        ## TODO: Rename function name
        scene_scale_factor = pose_utils.calculate_scene_scale(
            poses = all_poses, bounds = all_bounds, 
            bounds_method = self.params.preprocessing.bounds_method,
            intrinsics = all_intrinsics, height = H, width = W
        )

        ## TODO: Explain.
        adj_scale_factor = scene_scale_factor * self._scale_mul + self._scale_add

        for split in self.splits:
            data = data_splits[split]

            new_poses, new_bounds = pose_utils.reconfigure_scene_scale(
                old_poses = data.poses, old_bounds = data.bounds, 
                scene_scale_factor = adj_scale_factor,
            )
            new_data = ContainerType1(
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
        output = self._reconfigure_scene_scale(temp_data_step_2)

        return output

    def process_data(self, data):
        """
        Creates rays_o, rays_d, near, far, rgb

        TODO: Elaborate.

        Legend:
            N: Number of images.
            L: Total number of pixels. 

        TODO: Write Args and Returns
        """
        rays_o, rays_d = [], []
        near, far, rgb = [], [], []
        
        for idx in range(len(data.imgs)):
            
            img = data.imgs[idx]
            K = data.intrinsics[idx]

            H, W = img.shape[:2]
            rays_o_, rays_d_ = ray_utils.get_rays(
                H = H, W = W, intrinsic = K, c2w = data.poses[idx],
            )
            
            # Diving by 255 so that the range of the rgb 
            # data is between [0, 1]
            rgb_ = img.reshape(-1, 3).astype(np.float32)
            rgb_ = rgb_ / 255

            rgb.append(rgb_)
            rays_o.append(rays_o_)
            rays_d.append(rays_d_)

            bounds_ = data.bounds[idx]
            bounds_ = np.broadcast_to(
                bounds_[None, :], shape = (rays_d_.shape[0], 2)
            )
            near_, far_ = bounds_[:, 0:1], bounds_[:, 1:2]

            near.append(near_)
            far.append(far_)

        rgb = np.concatenate(rgb, axis = 0).astype(np.float32)
        far = np.concatenate(far, axis = 0).astype(np.float32)
        near = np.concatenate(near, axis = 0).astype(np.float32)
        rays_o = np.concatenate(rays_o, axis = 0).astype(np.float32)
        rays_d = np.concatenate(rays_d, axis = 0).astype(np.float32)

        data = ContainerType2(
            rays_o = rays_o, rays_d = rays_d, near = near, 
            far = far, rgb = rgb,
        )

        return data

    def prepare_data(self, data_splits):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        """
        processed_splits = {}
        for split in self.splits:
            reconf_data_splits = self.validate_and_reconfigure_data(
                data_splits = data_splits,
            )
            processed_splits[split] = self.process_data(reconf_data_splits[split])

        # img_HW holds the height and width of the images in the 
        # dataset. Since all the images in the dataset will have the 
        # same shape, we just get the shape of the first image 
        # in reconf_data_splits["train"]
        img_HW = reconf_data_splits["train"].imgs[0].shape[:2]
            
        return processed_splits, img_HW

    def _shuffle(self, container_type_2):
        """
        TODO: Docstring.
        """
        rng = np.random.default_rng(
            seed = self.params.data.train_shuffle.seed
        )
        perm = rng.permutation(len(container_type_2.rays_o))

        shuffled_rays_o = container_type_2.rays_o[perm]
        shuffled_rays_d = container_type_2.rays_d[perm]
        shuffled_near = container_type_2.near[perm]
        shuffled_far = container_type_2.far[perm]
        shuffled_rgb = container_type_2.rgb[perm]

        shuffled_container_type_2 = ContainerType2(
            rays_o = shuffled_rays_o, rays_d = shuffled_rays_d,
            near = shuffled_near, far = shuffled_far,
            rgb = shuffled_rgb,
        )

        return shuffled_container_type_2

    def _separate(self, container_type_2):
        """
        TODO: Elaborate.
        """
        CT2 = container_type_2
        x_split = (CT2.rays_o, CT2.rays_d, CT2.near, CT2.far)
        y_split = (CT2.rgb,)

        return x_split, y_split

    def create_tf_dataset(self, processed_splits):
        """
        Method that can be used by the subclasses.
        
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
        
        if self.params.data.train_shuffle.enable:
            processed_splits["train"] = self._shuffle(processed_splits["train"])

        x_train, y_train = self._separate(processed_splits["train"])
        x_test, y_test = self._separate(processed_splits["test"])
        x_val, y_val = self._separate(processed_splits["val"])

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

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        ## TODO: Return spec of some sort.
        tf_datasets = {
            "train": train_dataset, 
            "test": test_dataset,
            "val": val_dataset,
        }
        logger.debug("Created TensorFlow datasets.")

        return tf_datasets

