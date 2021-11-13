import os
import cv2
import json
import yaml
import imageio
import logging
import numpy as np
import pandas as pd

from nerf.utils import pose_utils
from nerf.core.base_dataset import Dataset, SceneLevelData

# Setting up logger.
logger = logging.getLogger(__name__)

class BlenderDataset(Dataset):
    """
    Class for handling the synthetic blender datasets.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.white_bg = self.params.system.white_bg
        self.data_params = self.params.data
        
        self.blender_dataset_params = self.params.blender_dataset
        self.root = self.blender_dataset_params.base_dir

        self._configure_dataset()

    @staticmethod
    def _return_sorted_paths(folder, meta):
        """
        Returns the list of paths of the desired images in the given 
        folder in a sorted order. 

        The paths are ordered in increasing order by the number in the 
        filename of the image.
        """
        paths, numbers = [], []
        
        for frame in meta["frames"]:
            number = int(frame["file_path"].split("/")[-1].split("_")[-1].strip())
            numbers.append(number)

        numbers = sorted(numbers)
        paths = [os.path.join(folder, f"r_{x}.png") for x in numbers]

        return paths

    @staticmethod
    def _process_metadata(metadata):
        """
        Organizes some information from the metadata so that it 
        is easier to process it downstream.
        """
        new_metadata = dict()
        new_metadata["camera_angle_x"] = metadata["camera_angle_x"]

        for frame in metadata["frames"]:
            name = frame["file_path"].split("/")[-1]
            new_metadata[name] = dict()
            new_metadata[name]["rotation"] = frame["rotation"]
            new_metadata[name]["transform_matrix"] = frame["transform_matrix"]

        return new_metadata

    @staticmethod
    def _opengl_to_classic_cv(pose):
        """
        Transforms a pose in OpenGL format to the Classic CV format. For 
        information about the formats, please refer to the documentation.
        """
        transformation = np.array([
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ], dtype = np.float64)
        output = pose @ transformation

        return output

    @staticmethod
    def _create_intrinsic_matrix(H, W, focal_length):
        """
        Creates an intrinsic matrix.

        TODO: Verify
        """
        fx, fy = focal_length, focal_length
        cx, cy = W/2, H/2
        
        intrinsic = np.array([
            [ fx, 0.0,  cx],
            [0.0,  fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype = np.float64)

        return intrinsic

    def _configure_dataset(self):
        """
        This function loads the metadata and sets up the paths to the images.
        """
        splits = ["train", "test", "val"]
        self.img_paths, self.metadata = dict(), dict()
        shuffle_params = self.blender_dataset_params.shuffle
        
        for split in splits:
            img_folder = os.path.join(self.root, split)
            json_path = os.path.join(self.root, f"transforms_{split}.json")
            
            with open(json_path, "r") as file:
                original_metadata = json.load(file)

            paths = self._return_sorted_paths(img_folder, original_metadata)
            self.metadata[split] = self._process_metadata(original_metadata)

            # Shuffling the paths if shuffle is enabled.
            if split in shuffle_params.enable:
                temp = np.array(paths)

                # Shuffling the paths.
                rng = np.random.default_rng(seed = shuffle_params.seed)
                perm = rng.permutation(len(paths))
                temp = temp[perm]
                
                # Overwritting paths with the shuffled paths.
                paths = temp.tolist()

            if split in ("test", "val"):
                split_params = getattr(self.blender_dataset_params, split)
                frac, num = split_params.frac, split_params.num

                if (frac is not None) and (num is None):
                    assert (type(frac) is float) and (frac > 0) and (frac < 1)
                    num_paths = int(len(paths) * frac)

                    assert num_paths <= len(paths)
                    paths = paths[:num_paths]

                elif (frac is None) and (num is not None):
                    assert type(num) is int
                    num_paths = num

                    assert num_paths <= len(paths)
                    paths = paths[:num_paths]

                elif (frac is None) and (num is None):
                    # Nothing to be done in this case. All the paths 
                    # will be used.
                    pass

                elif (frac is not None) and (num is not None):
                    ## TODO: redo message.
                    raise ValueError(
                        "Either frac could be not None, OR num "
                        "could be not None."
                    )

            self.img_paths[split] = paths

    def _load_split(self, split):
        """
        Loads the data for the given split.

        Args:
            split   :   A string which can be one among "train", "val" or "test".

        Returns:
            data    :   An object of type SceneLevelData containing the loaded data 
                        for the given split.
        """
        imgs, poses, bounds = [], [], []

        for path in self.img_paths[split]:
            name = os.path.basename(path).split(".")[0]

            if self.white_bg:
                temp = imageio.imread(path, "PNG-PIL").astype(np.float32)
                rgb = temp[..., :3]
                alpha = temp[..., 3] / 255.0
                img = rgb * alpha[..., None] + 255.0 * (1 - alpha[..., None])
                img = img.astype(np.uint8)
                
            else:
                img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)

            opengl_pose = self.metadata[split][name]["transform_matrix"]
            classic_cv_pose = self._opengl_to_classic_cv(opengl_pose)

            # For the blender dataset, for all images, near bound 
            # is 2.0 and far bound is 6.0
            bound = [2.0, 6.0]
            
            imgs.append(img)
            poses.append(classic_cv_pose)
            bounds.append(bound)

        # TODO: Review
        H, W = imgs[0].shape[:2]
        cam_angle_x = self.metadata[split]['camera_angle_x']
        focal = 0.5 * W / np.tan(0.5 * float(cam_angle_x))

        intrinsic = self._create_intrinsic_matrix(H, W, focal)
        intrinsics = [intrinsic.copy() for _ in range(len(imgs))]

        imgs = np.array(imgs).astype(np.uint8)
        poses = np.array(poses).astype(np.float32)
        bounds = np.array(bounds).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)

        data = SceneLevelData(
            imgs = imgs, poses = poses, 
            bounds = bounds, intrinsics = intrinsics
        )

        return data

    def get_data_and_metadata_for_splits(self):
        """
        This function returns data_splits and num_imgs for the blender dataset.

        This function overrides the abstractmethod get_data_and_metadata_for_splits 
        which is in the class Dataset.

        This variable num_imgs is needed because the number of images 
        that are available for each split will only be known at runtime. 
        For instance, the number of images available for validation may be 
        lesser than the total number of validation images depending on certain 
        parameters in the config file. To let the code accurately know 
        about the exact number of images available for each split, this 
        variable needs to be configured.

        Args:
            split       :   A string which can be one among "train", "val" or "test".

        Returns:
            data_splits :   A dictionary. Each key of this dictionary should be 
                            a string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an object of type 
                            SceneLevelData which contains the data for that split.

            num_imgs    :   A dictionary. Each key of this dictionary should be a 
                            string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an integer denoting 
                            the number of images that are available for the 
                            corresponding split.
        """
        logger.debug("Loading blender dataset.")
        
        num_imgs, data_splits = dict(), dict()
        splits = ["train", "test", "val"]

        for split in splits:
            data_splits[split] = self._load_split(split = split)
            num_imgs[split] = len(self.img_paths[split])
    
        logger.debug("Loaded blender dataset.")

        return data_splits, num_imgs

    def get_tf_datasets_and_metadata_for_splits(self):
        """
        This function returns tf_datasets, num_imgs and img_HW for the blender dataset.

        This function overrides the abstractmethod get_tf_datasets_and_metadata_for_splits 
        which is in the class Dataset.

        This variable num_imgs is needed because the number of images 
        that are available for each split will only be known at runtime. 
        For instance, the number of images available for validation may be 
        lesser than the total number of validation images depending on certain 
        parameters in the config file. To let the code accurately know 
        about the exact number of images available for each split, this 
        variable needs to be configured.

        The variable img_HW is returned to let the code accurately know about 
        the height and width of any processed image.

        Args:
            split       :   A string which can be one among "train", "val" or "test".

        Returns:
            tf_datasets :   A dictionary. Each key of this dictionary should be 
                            a string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be the TF Dataset object 
                            for that split.

            num_imgs    :   A dictionary. Each key of this dictionary should be a 
                            string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an integer denoting 
                            the number of images that are available for the 
                            corresponding split.

            img_HW      :   A tuple. img_HW[0] denotes the height of any processed 
                            image of the dataset and img_HW[1] denotes the width of 
                            any processed image of the dataset.
        """
        data_splits, num_imgs = self.get_data_and_metadata_for_splits()

        if self.params.data.dataset_mode == "iterate":
            prepared_splits, img_HW = self.prepare_data_iterate_mode(data_splits)
            tf_datasets = self.create_tf_datasets_iterate_mode(prepared_splits)

        elif self.params.data.dataset_mode == "sample":
            prepared_splits, img_HW = self.prepare_data_sample_mode(data_splits)
            tf_datasets = self.create_tf_datasets_sample_mode(prepared_splits)

        return tf_datasets, num_imgs, img_HW

class CustomDataset(Dataset):
    """
    Class for handling a custom dataset.
    """
    ## Defining some camera model related class attributes which will 
    ## be useful for some functions.

    # Camera models are as defined in
    # https://github.com/colmap/colmap/blob/8bfff64843aea6c648ed7dfa5c28dd5b9d766b3b/src/base/camera_models.h
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
        self.custom_dataset_params = self.params.custom_dataset
        self._configure_dataset()

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

        Args:
            camera_model    :   A string denoting the camera model. The list of supported 
                                camera models are mentioned in the class variable 
                                SUPPORTED_CAMERA_MODELS
            model_params    :   A list or a NumPy array with the parameters for the 
                                specified camera model.

        Returns:
            intrinsic       :   A NumPy array of shape (3, 3) with the intrinsic matrix.
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

    def _configure_dataset(self):
        """
        This function loads the metadata and sets up the paths to the images.
        """
        splits = ["train", "test", "val"]
        self.metadata = dict()
        shuffle_params = self.custom_dataset_params.shuffle
        
        for split in splits:
            split_params = self.custom_dataset_params[split]
            csv_path = split_params.pose_info_path
            df = pd.read_csv(csv_path)

            # Shuffling the DataFrame if shuffle is enabled.
            if split in shuffle_params.enable:
                df = df.sample(frac = 1, random_state = shuffle_params.seed)
                df = df.reset_index(drop = True)

            if split in ("test", "val"):
                frac, num = split_params.frac, split_params.num

                if (frac is not None) and (num is None):
                    assert (type(frac) is float) and (frac > 0) and (frac < 1)
                    num_rows = int(len(df) * frac)

                    assert num_rows <= len(df)
                    df = df.iloc[:num_rows]

                elif (frac is None) and (num is not None):
                    assert type(num) is int
                    num_rows = num

                    assert num_rows <= len(df)
                    df = df.iloc[:num_rows]

                elif (frac is None) and (num is None):
                    # Nothing to be done in this case. All the paths 
                    # will be used.
                    pass

                elif (frac is not None) and (num is not None):
                    ## TODO: redo message.
                    raise ValueError(
                        "Either frac could be not None, OR num "
                        "could be not None."
                    )

            self.metadata[split] = df

    def _load_split(self, split):
        """
        Loads the data for the given split.

        Args:
            split   :   A string which can be one among "train", "val" or "test".

        Returns:
            data    :   An object of type SceneLevelData containing the loaded data 
                        for the given split.
        """
        imgs, poses, bounds, intrinsics = [], [], [], []
        split_params = self.custom_dataset_params[split]

        for row in self.metadata[split].itertuples():
            
            path = os.path.join(split_params.img_root_dir, row.image_name)
            img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)

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

        imgs = np.array(imgs).astype(np.uint8)
        poses = np.array(poses).astype(np.float32)
        bounds = np.array(bounds).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)

        data = SceneLevelData(
            imgs = imgs, poses = poses, 
            bounds = bounds, intrinsics = intrinsics
        )

        return data

    def get_data_and_metadata_for_splits(self):
        """
        This function returns data_splits and num_imgs for the custom dataset.

        This function overrides the abstractmethod get_data_and_metadata_for_splits 
        which is in the class Dataset.

        This variable num_imgs is needed because the number of images 
        that are available for each split will only be known at runtime. 
        For instance, the number of images available for validation may be 
        lesser than the total number of validation images depending on certain 
        parameters in the config file. To let the code accurately know 
        about the exact number of images available for each split, this 
        variable needs to be configured.

        Args:
            split       :   A string which can be one among "train", "val" or "test".

        Returns:
            data_splits :   A dictionary. Each key of this dictionary should be 
                            a string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an object of type 
                            SceneLevelData which contains the data for that split.

            num_imgs    :   A dictionary. Each key of this dictionary should be a 
                            string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an integer denoting 
                            the number of images that are available for the 
                            corresponding split.
        """
        logger.debug("Loading custom dataset.")
        
        num_imgs, data_splits = dict(), dict()
        splits = ["train", "test", "val"]

        for split in splits:
            data_splits[split] = self._load_split(split = split)
            num_imgs[split] = len(data_splits[split].imgs)
    
        logger.debug("Loaded custom dataset.")

        return data_splits, num_imgs

    def get_tf_datasets_and_metadata_for_splits(self):
        """
        This function returns tf_datasets, num_imgs and img_HW for the custom dataset.

        This function overrides the abstractmethod get_tf_datasets_and_metadata_for_splits 
        which is in the class Dataset.

        This variable num_imgs is needed because the number of images 
        that are available for each split will only be known at runtime. 
        For instance, the number of images available for validation may be 
        lesser than the total number of validation images depending on certain 
        parameters in the config file. To let the code accurately know 
        about the exact number of images available for each split, this 
        variable needs to be configured.

        The variable img_HW is returned to let the code accurately know about 
        the height and width of any processed image.

        Args:
            split       :   A string which can be one among "train", "val" or "test".

        Returns:
            tf_datasets :   A dictionary. Each key of this dictionary should be 
                            a string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be the TF Dataset object 
                            for that split.

            num_imgs    :   A dictionary. Each key of this dictionary should be a 
                            string denoting a split (i.e. train/val/test). Each 
                            value of the dictionary should be an integer denoting 
                            the number of images that are available for the 
                            corresponding split.

            img_HW      :   A tuple. img_HW[0] denotes the height of any processed 
                            image of the dataset and img_HW[1] denotes the width of 
                            any processed image of the dataset.
        """
        data_splits, num_imgs = self.get_data_and_metadata_for_splits()

        if self.params.data.dataset_mode == "iterate":
            prepared_splits, img_HW = self.prepare_data_iterate_mode(data_splits)
            tf_datasets = self.create_tf_datasets_iterate_mode(prepared_splits)

        elif self.params.data.dataset_mode == "sample":
            prepared_splits, img_HW = self.prepare_data_sample_mode(data_splits)
            tf_datasets = self.create_tf_datasets_sample_mode(prepared_splits)

        return tf_datasets, num_imgs, img_HW

def get_dataset_obj(params):
    """
    Creates and returns an object of type BlenderDataset or CustomDataset 
    based on the setting of the dataset type in the config file.

    Args:
        params      :   The params object as returned by the function load_params.
                        The function load_params is defined in params_utils.py

    Returns:
        dataset_obj :   An object of BlenderDataset or CustomDataset based on
                        the setting of the dataset_type in the config file.
    """
    if params.system.dataset_type == "BlenderDataset":
        dataset_obj = BlenderDataset(params = params)
    elif params.system.dataset_type == "CustomDataset":
        dataset_obj = CustomDataset(params = params)
    else:
        raise ValueError(f"Invalid dataset type: {params.system.dataset_type}")

    if (type(dataset_obj) is not BlenderDataset) and \
        (params.system.white_bg):
        raise AssertionError("white_bg is only supported for BlenderDataset")

    return dataset_obj

def get_data_and_metadata_for_splits(params, return_dataset_obj = False):
    """
    Function that enables to user to get the data, num_imgs and optionaly 
    the dataset object for the desired dataset.

    If return_dataset_obj is False, then this function returns data_splits
    and num_imgs. If return_dataset_obj is True, then this function returns 
    data_splits, num_imgs and also the object of the desired dataset.

    This variable num_imgs is needed because the number of images 
    that are available for each split will only be known at runtime. 
    For instance, the number of images available for validation may be 
    lesser than the total number of validation images depending on certain 
    parameters in the config file. To let the code accurately know 
    about the exact number of images available for each split, this 
    variable needs to be configured.

    Args:
        params              :   The params object as returned by the function load_params. 
                                The function load_params is defined in params_utils.py

        return_dataset_obj  :   A boolean flag which determines if the object of the 
                                desired dataset is returned. Default is set to False. 
                                Please refer to the above explanation for more information 
                                about this parameter.

    Returns:
        A tuple named output. output[0] will always contain data_splits. output[1] will 
        always contain num_imgs. output[2] exists ONLY IF return_dataset_obj is True. 
        If return_dataset_obj is True, output[2] will contain dataset_obj. 
        If return_dataset_obj is False, trying to access output[2] will cause an 
        IndexError since output[2] does not exist in this case.

        Descriptions of the possible contents of the tuple output:

        data_splits         :   A dictionary. Each key of this dictionary should be 
                                a string denoting a split (i.e. train/val/test). Each 
                                value of the dictionary should be an object of type 
                                SceneLevelData which contains the data for that split.
                                This variable will be present in output[0].

        num_imgs            :   A dictionary. Each key of this dictionary should be a 
                                string denoting a split (i.e. train/val/test). Each 
                                value of the dictionary should be an integer denoting 
                                the number of images that are available for the 
                                corresponding split. This variable will be present 
                                in output[1].

        dataset_obj         :   An object of BlenderDataset or CustomDataset based on 
                                the setting of the dataset_type in the config file. 
                                This variable will be present in output[2] ONLY IF 
                                return_dataset_obj is True. If return_dataset_obj is False, 
                                trying to access output[2] will cause an IndexError since 
                                output[2] does not exist in this case.
    """
    dataset_obj = get_dataset_obj(params = params)
    data_splits, num_imgs = \
        dataset_obj.get_data_and_metadata_for_splits()

    if return_dataset_obj:
        output = (data_splits, num_imgs, dataset_obj)
    else:
        output = (data_splits, num_imgs)

    return output

def get_tf_datasets_and_metadata_for_splits(params, return_dataset_obj = False):
    """
    Function that enables to user to get the tf_datasets, num_imgs, img_HW 
    and optionaly the dataset object for the desired dataset.

    If return_dataset_obj is False, then this function returns tf_datasets, num_imgs 
    and img_HW. If return_dataset_obj is True, then this function returns 
    tf_datasets, num_imgs, img_HW and also the object of the desired dataset.

    This variable num_imgs is needed because the number of images 
    that are available for each split will only be known at runtime. 
    For instance, the number of images available for validation may be 
    lesser than the total number of validation images depending on certain 
    parameters in the config file. To let the code accurately know 
    about the exact number of images available for each split, this 
    variable needs to be configured.

    The variable img_HW is returned to let the code accurately know about 
    the height and width of any processed image.

    Args:
        params              :   The params object as returned by the function load_params. 
                                The function load_params is defined in params_utils.py

        return_dataset_obj  :   A boolean flag which determines if the object of the 
                                desired dataset is returned. Default is set to False. 
                                Please refer to the above explanation for more information 
                                about this parameter.

    Returns:
        A tuple named output. output[0] will always contain tf_datasets. 
        output[1] will always contain num_imgs. output[2] will always contain img_HW. 
        output[3] exists ONLY IF return_dataset_obj is True. If return_dataset_obj 
        is True, output[3] will contain dataset_obj. If return_dataset_obj is False, 
        trying to access output[3] will cause an IndexError since output[3] does not 
        exist in this case.

        tf_datasets         :   A dictionary. Each key of this dictionary should be 
                                a string denoting a split (i.e. train/val/test). Each 
                                value of the dictionary should be the TF Dataset object 
                                for that split. This variable will be present in output[0].

        num_imgs            :   A dictionary. Each key of this dictionary should be a 
                                string denoting a split (i.e. train/val/test). Each 
                                value of the dictionary should be an integer denoting 
                                the number of images that are available for the 
                                corresponding split. This variable will be present 
                                in output[1].

        img_HW              :   A tuple. img_HW[0] denotes the height of any processed 
                                image of the dataset and img_HW[1] denotes the width of 
                                any processed image of the dataset. This variable will 
                                be present in output[2].

        dataset_obj         :   An object of BlenderDataset or CustomDataset based on 
                                the setting of the dataset_type in the config file. 
                                This variable will be present in output[3] ONLY IF 
                                return_dataset_obj is True. If return_dataset_obj is False, 
                                trying to access output[3] will cause an IndexError since 
                                output[3] does not exist in this case.
    """
    dataset_obj = get_dataset_obj(params = params)
    tf_datasets, num_imgs, img_HW = \
        dataset_obj.get_tf_datasets_and_metadata_for_splits()

    if (params.data.dataset_mode == "iterate") and \
        (params.data.iterate_mode.advance_train_tf_dataset.enable):

        skip_count = params.data.iterate_mode.advance_train_tf_dataset.skip_count
        tf_datasets["train"] = tf_datasets["train"].skip(skip_count)

    if return_dataset_obj:
        output = (tf_datasets, num_imgs, img_HW, dataset_obj)
    else:
        output = (tf_datasets, num_imgs, img_HW)

    return output

if __name__ ==  "__main__":
    
    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)
    tf_datasets, num_imgs, img_HW = \
        get_tf_datasets_and_metadata_for_splits(params)
