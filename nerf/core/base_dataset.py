import os
import logging

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import namedtuple
from nerf.utils import ray_utils, pose_utils

# Setting up logger.
logger = logging.getLogger(__name__)

# Setting up SceneLevelData.
SceneLevelData = namedtuple(
    'SceneLevelData', ("imgs", "poses", "bounds", "intrinsics")
)

# Setting up RayLevelData.
RayLevelData = namedtuple(
    'RayLevelData', ("rays_o", "rays_d", "near", "far", "rgb")
)

class Dataset(ABC):
    """
    This class contains useful functionality that can be used by 
    other classes which inherit this class. This class also makes 
    sure that the derived classes has implemented certain methods. 
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

    @abstractmethod
    def get_data_and_metadata_for_splits(self):
        """
        This method needs to be implemented by every class that inherits 
        this class (Dataset).

        The get_data_and_metadata_for_splits function implemented in each 
        subclass must return a tuple of two variables.

        The first variable of the tuple should be data_splits. This variable 
        should be a dictionary. Each key of this dictionary should be a string 
        denoting a split (i.e. train/val/test). Each value of the dictionary 
        should be an object of type SceneLevelData which contains the data 
        for that split.

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

        TODO: Refactor, review terminology, finish.
        """
        pass
    
    @abstractmethod
    def get_tf_datasets_and_metadata_for_splits(self):
        """
        This method needs to be implemented by every class that inherits
        this class (Dataset).

        The get_tf_datasets_and_metadata_for_splits function implemented in each 
        subclass must return a tuple of three variables.

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
            data_splits :   A dictionary. Each key of this dictionary should 
                            be a string denoting a split (i.e. train/val/test). 
                            Each value of the dictionary should be an object of 
                            type SceneLevelData which contains the loaded data 
                            for that split.
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

    def _save_reconfig_params(self, W1_to_W2_transform, adj_scale_factor):
        """
        Saves the parameters that were used for reconfiguring the data.

        Specifically, the parameters W1_to_W2_transform and 
        adj_scale_factor are saved to disk. These parameters can later
        be used during rendering.
        """
        to_save = {}
        root = self.params.data.reconfig.save_dir

        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        to_save["W1_to_W2_transform"] = W1_to_W2_transform
        to_save["adj_scale_factor"] = adj_scale_factor
        np.savez(os.path.join(root, "reconfig.npz"), **to_save)

    def load_reconfig_params(self):
        """
        Loads the reconfig parameters from an npz file. 
        """
        path = os.path.join(self.params.data.reconfig.save_dir, "reconfig.npz")
        data = np.load(path)
        
        W1_to_W2_transform = data["W1_to_W2_transform"]
        adj_scale_factor = data["adj_scale_factor"]

        return W1_to_W2_transform, adj_scale_factor

    def _reconfigure_imgs_and_intrinsics(self, data_splits):
        """
        Resizes the images and adjusts the intrinsics matrices if required 
        based on the settings in the config file.

        Args:
            data_splits         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

        Returns:
            reconf_data_splits  :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.
        """
        reconf_data_splits = dict()
        
        for split in self.splits:
            data = data_splits[split]
            
            new_imgs, new_intrinsics = pose_utils.scale_imgs_and_intrinsics(
                old_imgs = data.imgs, old_intrinsics = data.intrinsics, 
                scale_factor = self.params.data.scale_imgs,
            )

            reconf_data = SceneLevelData(
                imgs = new_imgs, poses = data.poses, 
                bounds = data.bounds, intrinsics = new_intrinsics,
            )
            reconf_data_splits[split] = reconf_data

        return reconf_data_splits

    def _reconfigure_poses(self, data_splits):
        """
        Reconfigures the pose matrices.

        The poses matrices of the input move a point from a camera coordinate system to 
        the W1 coordinate system. For each pose matrix in the input, this function 
        creates a new pose matrix that moves a point from a camera coordinate system 
        to the W2 coordinate system. (TODO: Rewrite.)

        Args:
            data_splits         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

        Returns:
            reconf_data_splits  :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

            W1_to_W2_transform  :   A NumPy array of shape (4, 4) which can transform a 
                                    point from the W1 corodinate system to the 
                                    W2 coordinate system.
        """
        reconf_data_splits = dict()
        
        all_poses = [data_splits[x].poses for x in self.splits]
        all_poses = np.concatenate(all_poses, axis = 0)

        W1_to_W2_transform = pose_utils.calculate_new_world_transform(
            poses = all_poses, 
            origin_method = self.params.preprocessing.origin_method,
            basis_method = self.params.preprocessing.basis_method,
            manual_rotation = self.params.preprocessing.manual_rotation,
        )

        for split in self.splits:
            data = data_splits[split]

            new_poses = pose_utils.reconfigure_poses(
                old_poses = data.poses, 
                W1_to_W2_transform = W1_to_W2_transform
            )
            new_data = SceneLevelData(
                imgs = data.imgs, poses = new_poses, 
                bounds = data.bounds, intrinsics = data.intrinsics,
            )
            reconf_data_splits[split] = new_data

        return reconf_data_splits, W1_to_W2_transform

    def _reconfigure_scene_scale(self, data_splits):
        """
        Reconfigures the pose matrices and bounds to adjust the scale of the 
        scene if required.

        This function calculates the scene scale factor (scene_scale_factor) 
        using the function pose_utils.calculate_scene_scale. The calculated 
        scene scale factor can be adjusted using certain parameters defined 
        in the config file. The adjusted scene scale factor (adj_scale_factor)
        is then used to reconfigure the poses matrices and bounds.

        Args:
            data_splits         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

        Returns:
            reconf_data_splits  :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

            adj_scale_factor    :   A float value (TODO: check if type is float or numpy)
                                    denoting the adjusted scale factor.

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

        scene_scale_factor = pose_utils.calculate_scene_scale(
            poses = all_poses, bounds = all_bounds, 
            bounds_method = self.params.preprocessing.bounds_method,
            intrinsics = all_intrinsics, height = H, width = W
        )

        # The values self._scale_mul and self._scale_add are used to 
        # adjust the scene scale factor. The adjusted scene scale 
        # factor (adj_scale_factor) is then used to reconfigure the 
        # poses matrices and bounds.
        adj_scale_factor = scene_scale_factor * self._scale_mul + self._scale_add

        for split in self.splits:
            data = data_splits[split]

            new_poses, new_bounds = pose_utils.reconfigure_scene_scale(
                old_poses = data.poses, old_bounds = data.bounds, 
                scene_scale_factor = adj_scale_factor,
            )
            new_data = SceneLevelData(
                imgs = data.imgs, poses = new_poses, 
                bounds = new_bounds, intrinsics = data.intrinsics,
            )
            reconf_data_splits[split] = new_data

        return reconf_data_splits, adj_scale_factor

    def validate_and_reconfigure_data(self, data_splits):
        """
        Validates and reconfigures the data.

        This function performs the following steps:
        1.  Checks if the data provided for the train, val and tests splits 
            is consistent with the requirements of the codebase.

        2.  Resizes the images and adjusts the intrinsics matrices if required 
            based on the settings in the config file.

        3.  Reconfigure pose matrices (TODO: rewrite)

        4.  Reconfigures the pose matrices and bounds to adjust the scale of the 
            scene if required.

        5.  Saves some of the metadata calculated during the above steps to 
            disk if required.

        Args:
            data_splits     :   A dictionary. Each key of this dictionary should 
                                be a string denoting a split (i.e. train/val/test). 
                                Each value of the dictionary should be an object of 
                                type SceneLevelData which contains the data for 
                                that split.

        Returns:
            output          :   A dictionary. Each key of this dictionary should 
                                be a string denoting a split (i.e. train/val/test). 
                                Each value of the dictionary should be an object of 
                                type SceneLevelData which contains the data for 
                                that split.
        """
        self._validate_all_splits(data_splits)

        temp_data_step_1 = self._reconfigure_imgs_and_intrinsics(data_splits)
        temp_data_step_2, W1_to_W2_transform = self._reconfigure_poses(temp_data_step_1)
        output, adj_scale_factor = self._reconfigure_scene_scale(temp_data_step_2)

        if self.params.data.reconfig.save_dir is not None:
            self._save_reconfig_params(W1_to_W2_transform, adj_scale_factor)

        return output

    def process_data(self, data):
        """
        Extracts ray level information from the given scene level data. 
    
        Given an object of type SceneLevelData which contains scene level data 
        (i.e. imgs, poses, bounds, intrinsics), this function extracts ray level 
        data (i.e. rays_o, rays_d, near, far, rgb). The extracted ray level data 
        is stored in an object of type RayLevelData.

        TODO: Describe shapes of rays_o etc.

        Args:
            data    :   An object of type SceneLevelData

        Returns:
            output  :   An object of type RayLevelData
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

        output = RayLevelData(
            rays_o = rays_o, rays_d = rays_d, near = near, 
            far = far, rgb = rgb,
        )

        return output

    def prepare_data_iterate_mode(self, data_splits):
        """
        This function prepares the data for the iterate dataset mode. 
        This function can be used by the subclasses.

        Args:
            data_splits         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

        Returns:
            processed_splits    :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type RayLevelData which contains the data for 
                                    that split.

            img_HW              :   A tuple denoting the height and width of the images 
                                    in the dataset (all images in the dataset have the same 
                                    height and width). img_HW[0] contains the height and 
                                    img_HW[1] contains the width.
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
        This function prepares the data for the sample dataset mode. 
        This function can be used by the subclasses.

        Args:
            data_splits         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type SceneLevelData which contains the data for 
                                    that split.

        Returns:
            processed_splits    :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    processed_splits["train"] will contain an object of 
                                    type SceneLevelData. processed_splits["val"] and 
                                    processed_splits["test"] will contain an object 
                                    of type RayLevelData.

            img_HW              :   A tuple denoting the height and width of the images 
                                    in the dataset (all images in the dataset have the same 
                                    height and width). img_HW[0] contains the height and 
                                    img_HW[1] contains the width.
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
        This is a function that will be used by train_dataset.map 
        in the function create_tf_datasets_sample_mode

        This function extracts ray level data from the a given image, 
        pose matrix, bounds and intrinsic matrix. If the image has
        height H and width W, then H*W rays can be obtained.

        The ray level data is comprised of ray origin, ray direction, 
        near bound, far bound and rgb colour information for each ray.

        This function randomly (uniformly) selects B rays among 
        the H*W available rays and returns the corresponding ray 
        level information for the B rays. Here, B is the batch size.

        TODO: Verify type (TF Tensor) and shapes.

        Args:
            img         :   A TensorFlow tensor of shape (H, W, 3)
            pose        :   A TensorFlow tensor of shape (4, 4)
            bounds      :   A TensorFlow tensor of shape (2,)
            intrinsic   :   A TensorFlow tensor of shape (3, 3)

        Returns:
            x_vals      :   A tuple containing rays_o, rays_d, near 
                            and far. The shape of rays_o and rays_d 
                            is (B, 3). The shape of near and far 
                            is (B, 1).
            y_vals      :   A tuple containing rgb. The shape of 
                            rgb is (B, 3)
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

    def _shuffle(self, ray_level_data):
        """
        Given an object of type RayLevelData (here, ray_level_data),
        this function performs the following operations:

        1.  Shuffles all the contents of ray_level_data
        2.  Creates a new instance of RayLevelData called 
            shuffled_ray_level_data to store the shuffled contents. 
        """
        rng = np.random.default_rng(
            seed = self.iterate_mode_params.train_shuffle.seed
        )
        perm = rng.permutation(len(ray_level_data.rays_o))

        shuffled_rays_o = ray_level_data.rays_o[perm]
        shuffled_rays_d = ray_level_data.rays_d[perm]
        shuffled_near = ray_level_data.near[perm]
        shuffled_far = ray_level_data.far[perm]
        shuffled_rgb = ray_level_data.rgb[perm]

        shuffled_ray_level_data = RayLevelData(
            rays_o = shuffled_rays_o, rays_d = shuffled_rays_d,
            near = shuffled_near, far = shuffled_far,
            rgb = shuffled_rgb,
        )

        return shuffled_ray_level_data

    def _separate(self, ray_level_data):
        """
        Given an object of type RayLevelData (here, ray_level_data), 
        this function splits the contents of the object into 
        x_split and y_split. 
        """
        RLD = ray_level_data
        x_split = (RLD.rays_o, RLD.rays_d, RLD.near, RLD.far)
        y_split = (RLD.rgb,)

        return x_split, y_split

    def create_tf_datasets_iterate_mode(self, processed_splits):
        """
        Creates TensorFlow datasets for the iterate dataset mode.

        Given ray level data for each split, this function creates 
        a TensorFlow (TF) dataset for each split. This function can 
        be used by the subclasses.
        
        Args:
            processed_splits    :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be an object of 
                                    type RayLevelData which contains the data for 
                                    that split.

        Returns:
            tf_datasets         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be a TF dataset.
        """
        logger.debug("Creating TensorFlow datasets.")

        # The user can optionally request the train data to be shuffled 
        # before the train TF dataset is created.
        if self.iterate_mode_params.train_shuffle.enable:
            processed_splits["train"] = self._shuffle(processed_splits["train"])

        x_train, y_train = self._separate(processed_splits["train"])
        x_test, y_test = self._separate(processed_splits["test"])
        x_val, y_val = self._separate(processed_splits["val"])

        # Creating the train TF dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = True,
        )
        train_dataset = train_dataset.repeat(
            count = self.iterate_mode_params.repeat_count
        )

        # Creating the val TF dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        # Creating the test TF dataset.
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

    def create_tf_datasets_sample_mode(self, processed_splits):
        """
        Creates TensorFlow datasets for the sample dataset mode.

        Given validated and reconfigured scene level data for the train split, 
        and ray level data for the val and test splits, this function creates 
        a TensorFlow (TF) dataset for each split. This function can be used 
        by the subclasses.
        
        Args:
            processed_splits    :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    processed_splits["train"] will contain an object of 
                                    type SceneLevelData. processed_splits["val"] and 
                                    processed_splits["test"] will contain an object 
                                    of type RayLevelData.

        Returns:
            tf_datasets         :   A dictionary. Each key of this dictionary should 
                                    be a string denoting a split (i.e. train/val/test). 
                                    Each value of the dictionary should be a TF dataset.
        """
        logger.debug("Creating TensorFlow datasets.")

        train_imgs = processed_splits["train"].imgs.astype(np.uint8)
        train_poses = processed_splits["train"].poses.astype(np.float32)
        train_bounds = processed_splits["train"].bounds.astype(np.float32)
        train_intrinsics = processed_splits["train"].intrinsics.astype(np.float32)

        x_test, y_test = self._separate(processed_splits["test"])
        x_val, y_val = self._separate(processed_splits["val"])

        # Creating the train TF dataset.
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

        # Creating the val TF dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        # Creating the test TF dataset.
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

    def create_dataset_for_render(self, H, W, c2w, bounds, intrinsic, reconfig_poses):
        """
        Create a TensorFlow dataset that can be used for rendering a single image. 
        
        TODO: Maybe elaborate; also check types and shapes.

        This function can be used by the subclasses.

        Args:
            H               :   An integer representing the height of the image.
            W               :   An integer representing the width of the image.
            c2w             :   A NumPy array of shape (4, 4) representing the
                                camera to world transformation matrix. The "world"
                                here could refer to the W1 or W2 coordinate system.
                                The choice of W1 or W2 is determined by the
                                reconfig_poses argument.
            bounds          :   A NumPy array of shape (2,) representing the 
                                near and far bounds.
            intrinsic       :   A NumPy array of shape (3, 3) representing the 
                                intrinsic matrix.
            reconfig_poses  :   A boolean which denotes if poses have to be
                                reconfigured. If True, c2w is assumed to be a
                                camera to W1 transformation matrix. If False,
                                c2w is assumed to be a camera to W2 
                                transformation matrix.

        Returns:
            dataset         :   A TensorFlow dataset.
        """
        self._validate_intrinsic_matrix(K = intrinsic)
        W1_to_W2_transform, adj_scale_factor = self.load_reconfig_params()

        if reconfig_poses:
            temp_pose = pose_utils.reconfigure_poses(
                old_poses = c2w, 
                W1_to_W2_transform = W1_to_W2_transform
            )
        else:
            temp_pose = c2w

        new_pose, new_bounds = pose_utils.reconfigure_scene_scale(
            old_poses = temp_pose, old_bounds = bounds, 
            scene_scale_factor = adj_scale_factor,
        )

        rays_o, rays_d = ray_utils.get_rays(
            H = H, W = W, intrinsic = intrinsic, c2w = new_pose,
        )

        bounds_ = np.broadcast_to(new_bounds[None, :], shape = (rays_d.shape[0], 2))
        near, far = bounds_[:, 0:1], bounds_[:, 1:2]

        far = far.astype(np.float32)
        near = near.astype(np.float32)
        rays_o = rays_o.astype(np.float32)
        rays_d = rays_d.astype(np.float32)

        data = (rays_o, rays_d, near, far)
        dataset = tf.data.Dataset.from_tensor_slices((data,))
        dataset = dataset.batch(
            batch_size =  self.params.data.batch_size,
            drop_remainder = False,
        )

        return dataset
