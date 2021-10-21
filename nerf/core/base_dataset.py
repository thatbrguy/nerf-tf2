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

        assert self.params.data.dataset_mode in ("sample", "iterate")
        self.sample_mode_params = self.params.data.sample_mode
        self.iterate_mode_params = self.params.data.iterate_mode

        # self._W2_to_W1_transform and self._adj_scale_factor are only used 
        # during inference. They are initialized to None here. Their actual 
        # values will be populated by the function _load_reconfig_params
        self._W2_to_W1_transform = None
        self._adj_scale_factor = None

    @abstractmethod
    def get_dataset(self):
        """
        This method needs to be implemented by every subclass.

        The get_dataset function implemented in each subclass must 
        return a tuple of three variables.

        The first variable of the tuple should be tf_datasets. This variable 
        should be a dictionary. Each key of this dictionary should be a string 
        denoting a split (i.e. train/val/test). Each value of the dictionary 
        should be the TF Dataset object for the corresponding split. 

        The second variable of the tuple should be num_imgs. This variable 
        should be a dictionary. Each key of this dictionary should be a string 
        denoting a split (i.e. train/val/test). Each value of the dictionary 
        should be an integer denoting the number of images that are available 
        for the corresponding split.

        The second variable (num_imgs) is needed because the number of images 
        that are available for each split will only be known at runtime. 
        For instance, the number of images available for validation may be 
        lesser than the total number of validation images depending on certain 
        parameters in the config file. To let the code accurately know 
        about the exact number of images available for each split, this 
        variable needs to be configured.

        The third variable of the tuple should be img_HW. This variable 
        should be a tuple. img_HW[0] denotes the height of any processed 
        image of the dataset and img_HW[1] denotes the width of any processed 
        image of the dataset. The third variable is included to let the code
        accurately know about the height and width of any processed image.

        TODO: Refactor, review terminology, finish.
        """
        pass

    @staticmethod
    def _validate_intrinsic_matrix(K):
        """
        This function checks if the given intrinsic matrix is of 
        a valid form.
        
        The current codebase only supports intrinsic matrices that 
        are of the form:
        
        intrinsic = np.array([
            [fu, 0., cu],
            [0., fv, cv],
            [0., 0., 1.],
        ])

        Args:
            K   :   A NumPy array of shape (3, 3) representing the 
                    intrinsic matrix. (TODO: Verify)
        """
        zero_check = np.array([
            K[0, 1], K[1, 0], K[2, 0], K[2, 1],
        ])
        assert K[2, 2] == 1
        assert K.shape == (3, 3)
        assert np.all(zero_check == 0)

    def _validate_all_splits(self, data_splits):
        """
        Checks if the data provided for the train, val and tests splits 
        is consistent with the requirements of the codebase.

        More specifically, this function checks: 
            1.  If all the images across all the splits have the 
                same height and width
            2.  If all the intrinsic matrices across all the 
                splits are of the form that is deemed acceptable by 
                the function _validate_intrinsic_matrix

        Args:
            TODO
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

    def _save_reconfig_params(self, W2_to_W1_transform, adj_scale_factor):
        """
        Saves the parameters that were used for reconfiguring the data.

        Specifically, the parameters W2_to_W1_transform and 
        adj_scale_factor are saved to disk. These parameters can later
        be used during inference.
        """
        to_save = {}
        root = self.params.data.reconfig.save_dir

        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        to_save["W2_to_W1_transform"] = W2_to_W1_transform
        to_save["adj_scale_factor"] = adj_scale_factor
        np.savez(os.path.join(root, "reconfig.npz"), **to_save)

    def _load_reconfig_params(self, W2_to_W1_transform, adj_scale_factor):
        """
        Loads the parameters that are needed reconfiguring the data 
        during inference mode from an npz file. 
        """
        path = os.path.join(self.params.data.reconfig.save_dir, "reconfig.npz")
        data = np.load(path)
        
        self._W2_to_W1_transform = data["W2_to_W1_transform"]
        self._adj_scale_factor = data["adj_scale_factor"]

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
            basis_method = self.params.preprocessing.basis_method,
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

    def prepare_data_iterate_mode(self, data_splits):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        """
        processed_splits = {}
        reconf_data_splits = self.validate_and_reconfigure_data(
            data_splits = data_splits,
        )
        for split in self.splits:
            processed_splits[split] = self.process_data(reconf_data_splits[split])

        # img_HW holds the height and width of the images in the 
        # dataset. Since all the images in the dataset will have the 
        # same shape, we just get the shape of the first image 
        # in reconf_data_splits["train"]
        img_HW = reconf_data_splits["train"].imgs[0].shape[:2]
            
        return processed_splits, img_HW

    def prepare_data_sample_mode(self, data_splits):
        """
        Method that can be used by the subclasses.

        TODO: Elaborate.
        """
        out_data_splits = {}
        reconf_data_splits = self.validate_and_reconfigure_data(
            data_splits = data_splits,
        )
        for split in self.splits:
            
            if split == "train":
                out_data_splits[split] = reconf_data_splits[split]

            else:
                processed_split = self.process_data(reconf_data_splits[split])
                out_data_splits[split] = processed_split

        # img_HW holds the height and width of the images in the 
        # dataset. Since all the images in the dataset will have the 
        # same shape, we just get the shape of the first image 
        # in out_data_splits["train"]
        img_HW = out_data_splits["train"].imgs[0].shape[:2]
            
        return out_data_splits, img_HW

    def _sample_mode_map_function(self, img, pose, bounds, intrinsic):
        """
        TODO: Docstring.
        """            
        K = intrinsic        
        img_shape = tf.shape(img)
        H, W = img_shape[0], img_shape[1]

        _rays_o, _rays_d = ray_utils.get_rays_tf(
            H = H, W = W, intrinsic = K, c2w = pose,
        )
        
        # Diving by 255 so that the range of the rgb 
        # data is between [0, 1]
        _rgb = tf.reshape(img, (-1, 3))
        _rgb = tf.cast(_rgb, tf.float32)
        _rgb = _rgb / 255.0
        
        bounds = tf.broadcast_to(
            bounds[None, :], shape = (tf.shape(_rays_d)[0], 2)
        )
        _near, _far = bounds[:, 0:1], bounds[:, 1:2]

        total_rays = tf.shape(_rgb)[0]
        idxs = tf.random.uniform(
            (self.params.data.batch_size,), minval=0, 
            maxval=total_rays, dtype=tf.int32,
        )

        rays_o = tf.gather(_rays_o, idxs)
        rays_d = tf.gather(_rays_d, idxs)
        near = tf.gather(_near, idxs)
        far = tf.gather(_far, idxs)
        rgb = tf.gather(_rgb, idxs)
        
        x_vals = (rays_o, rays_d, near, far)
        y_vals = (rgb,)
        
        return x_vals, y_vals

    def _shuffle(self, container_type_2):
        """
        Given an instance of ContainerType2 (which is container_type_2), 
        this function performs the following operations:

        1.  Shuffles all the contents of container_type_2
        2.  Creates a new instances of ContainerType2 called 
            shuffled_container_type_2 to store the shuffled contents. 
        """
        rng = np.random.default_rng(
            seed = self.iterate_mode_params.train_shuffle.seed
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
        Given an instance of ContainerType2 (which is container_type_2), 
        this function splits the contents of the instance into 
        x_split and y_split. 
        """
        CT2 = container_type_2
        x_split = (CT2.rays_o, CT2.rays_d, CT2.near, CT2.far)
        y_split = (CT2.rgb,)

        return x_split, y_split

    def create_tf_dataset_iterate_mode(self, processed_splits):
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
        
        if self.iterate_mode_params.train_shuffle.enable:
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
            count = self.iterate_mode_params.repeat_count
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

        tf_datasets = {
            "train": train_dataset, 
            "test": test_dataset,
            "val": val_dataset,
        }
        logger.debug("Created TensorFlow datasets.")

        return tf_datasets

    def create_tf_dataset_sample_mode(self, processed_splits):
        """
        Method that can be used by the subclasses.
        """
        logger.debug("Creating TensorFlow datasets.")

        train_imgs = processed_splits["train"].imgs
        train_poses = processed_splits["train"].poses.astype(np.float32)
        train_bounds = processed_splits["train"].bounds.astype(np.float32)
        train_intrinsics = processed_splits["train"].intrinsics.astype(np.float32)

        x_test, y_test = self._separate(processed_splits["test"])
        x_val, y_val = self._separate(processed_splits["val"])

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_imgs, train_poses, train_bounds, train_intrinsics)
        )
        train_dataset = train_dataset.repeat(
            count = self.sample_mode_params.repeat_count
        )
        train_dataset = train_dataset.shuffle(
            buffer_size = self.sample_mode_params.shuffle_buffer_size
        )
        train_dataset = train_dataset.map(self._sample_mode_map_function)
        train_dataset = train_dataset.prefetch(
            buffer_size = self.sample_mode_params.prefetch_buffer_size
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

        tf_datasets = {
            "train": train_dataset, 
            "test": test_dataset,
            "val": val_dataset,
        }
        logger.debug("Created TensorFlow datasets.")

        return tf_datasets
