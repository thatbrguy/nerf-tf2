import os
import cv2
import json
import yaml
import logging
import numpy as np
import pandas as pd

from nerf.utils import pose_utils
from nerf.core.base_dataset import Dataset, ContainerType1

# Setting up logger.
logger = logging.getLogger(__name__)

class BlenderDataset(Dataset):
    """
    Class for handling the synthetic blender datasets.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
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

        TODO: Elaborate.
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
        TODO: Docstring.
        """
        imgs, poses, bounds = [], [], []

        for path in self.img_paths[split]:
            name = os.path.basename(path).split(".")[0]

            img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
            opengl_pose = self.metadata[split][name]["transform_matrix"]
            classic_cv_pose = self._opengl_to_classic_cv(opengl_pose)

            # For the blender dataset, for all images, near bound 
            # is 2.0 and far bound is 6.0
            bound = [2.0, 6.0]
            
            imgs.append(img)
            poses.append(classic_cv_pose)
            bounds.append(bound)

        H, W = imgs[0].shape[:2]
        cam_angle_x = self.metadata[split]['camera_angle_x']
        focal = 0.5 * W / np.tan(0.5 * float(cam_angle_x))

        intrinsic = self._create_intrinsic_matrix(H, W, focal)
        intrinsics = [intrinsic.copy() for _ in range(len(imgs))]

        imgs = np.array(imgs)
        poses = np.array(poses)
        bounds = np.array(bounds)
        intrinsics = np.array(intrinsics)

        data = ContainerType1(
            imgs = imgs, poses = poses, 
            bounds = bounds, intrinsics = intrinsics
        )

        return data

    def _load_full_dataset(self):
        """
        TODO: Elaborate.
        """
        logger.debug("Loading blender dataset.")
        
        num_imgs, data_splits = dict(), dict()
        splits = ["train", "test", "val"]

        for split in splits:
            data_splits[split] = self._load_split(split = split)
            num_imgs[split] = len(self.img_paths[split])
    
        logger.debug("Loaded blender dataset.")

        return data_splits, num_imgs

    def get_dataset(self):
        """
        TODO: Elaborate.
        """
        data_splits, num_imgs = self._load_full_dataset()
        processed_splits, img_HW = super().prepare_data(data_splits)
        
        tf_datasets = super().create_tf_dataset(processed_splits)

        return tf_datasets, num_imgs, img_HW

class CustomDataset(Dataset):
    """
    Class for handling a custom dataset.

    NOTE: CustomDataset is currently not supported! Please do not 
    use CustomDataset until further notice.
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
        logger.warn(
            "CustomDataset is currently not supported! Please "
            "do not use CustomDataset until further notice."
        )
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
        logger.warn(
            "CustomDataset is currently not supported! Please "
            "do not use CustomDataset until further notice."
        )
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
        logger.warn(
            "CustomDataset is currently not supported! Please "
            "do not use CustomDataset until further notice."
        )
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

    def get_reconfigured_data(self):
        """
        TODO: Elaborate.
        """
        logger.warn(
            "CustomDataset is currently not supported! Please "
            "do not use CustomDataset until further notice."
        )
        (
            imgs, poses, 
            bounds, intrinsics
        ) = self._load_full_dataset()

        output = super().validate_and_reconfigure_data(
            imgs = imgs, poses = poses, 
            bounds = bounds, intrinsics = intrinsics,
        )

        return output

if __name__ ==  "__main__":
    pass
