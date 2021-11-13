# Data Preparation

This codebase currently supports two types of datasets:
1. The synthetic blender datasets which are mentioned in the official implementation.
2. Custom datasets which represent a "360-degree inward-facing" scene.

The following section elaborates how to prepare the dataset for the about two types.

## 1. Blender Dataset
The following blender datasets can be accessed through a link in the GitHub repository of the [official implementation](https://github.com/bmild/nerf#running-code): `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic` and `ship`.  

Setting up a blender dataset is very simple. Please follow the below instructions:

1. Download the desired blender dataset from the GitHub repository of the [official implementation](https://github.com/bmild/nerf#running-code). 
	- For example, if you are interested in the `lego` dataset, please download the entire `lego` folder which should be available inside the folder `nerf_synthetic`
2. Place your desired blender dataset somewhere on your local disk and make note of the path to your blender dataset.
	- For example, if you are interested in the `lego` blender dataset, and you have downloaded and placed the lego dataset at `/path/to/datasets`, then the path the `lego` dataset would be `/path/to/datasets/lego`
3. Set the parameters in `nerf/params/config.yaml` appropriately by following the instructions given above each parameter in the file.

## 2. 360-Degree Inward-Facing Custom Dataset
Setting up a custom dataset for a 360-degree inward-facing scene is more involved. This section describes the data preparation process for this dataset type.

### 2.1 Introduction
For any custom 360-degree inward-facing dataset, we require images, poses, bounds (near and far bounds) and intrinsics. Collectively, we will refer to the images, poses, bounds and intrinsics as "information about the scene".

In one case, the user may only have the images, and may not have the remaining information about the scene. In another case, the user may already have prepared all the information about the scene. This section describes how to prepare the data for **both** cases.

The codebase requires the information about the scene to be structured in a specific format. We will refer to this format as the "360-degree inward-facing custom dataset format" or "**IFCD Format**" in short. The **IFCD format** is as follows:
1. All the images should be stored in a single directory.
2. The information regarding the poses, bounds and intrinsics should be organized in a CSV file in the **Pose Info Format** (specified in section 2.4).

### 2.2 Setting up the information about the scene

In some cases, the user may only have the images, and may not have the remaining information about the scene (let us call this **Case 1**). In another case, the user may already have prepared all the information about the scene (let us call this **Case 2**). Let us explore how to set up the information about the scene for each case separately.

#### 2.2.1 Case 1
In this case, the user only has the images and does not have the remaining information about the scene. In this case, the user has to rely on some remaining codebase to estimate the remaining information about the scene.

In this case, the user can consider using [colmap-utils](https://github.com/thatbrguy/colmap_utils) for estimating the remaining information about the scene. 
- The remaining information about the scene estimated via [colmap-utils](https://github.com/thatbrguy/colmap_utils) will be in the Pose Info Format (section 2.4) and hence the format would be compatible with this codebase.
- However, do note that only a subset of the camera models defined in COLMAP are supported by this codebase (please check section 2.4 for more information). So do make sure the data estimated via [colmap-utils](https://github.com/thatbrguy/colmap_utils) is also compatible with this codebase (even though the format is compatible) before using the data.

If the user wishes to use an alternative approach to estimate the remaining information about the scene, they are free to do so. In that case, once the remaining information about the scene is estimated, the user should follow the instructions in **Case 2**. This is because, the estimated remaining information about the scene may not be in the Pose Info Format (section 2.4) that is required.

#### 2.2.2 Case 2
In this case, the user has already prepared all the information about the scene. In this case, the user should: 
1. Place all the images in one directory. This is done to satisfy the first requirement of the IFCD Format.
2. Organize the remaining information about the scene (poses, intrinsics, bounds) as per the format in section 2.4 (Pose Info Format). This is done to satisfy the second requirement of the IFCD format.

### 2.3 Creating train, val and test splits
Once all the required data is organized in the IFCD Format, the user should split the data into train, val and test splits. 

A convenience script is provided in this codebase to perform a random split of the data into train, val and test splits. The user can either use that script to create the splits, or can use their own custom method to create the splits. Do note that each split must have data which is in the IFCD format. 

The convenience script is `nerf/utils/data_utils.py`. Instructions on using the convenience script is as follows:

- Follow the procedure mentioned in the common steps section of the `README.md` file (section 5.1 of `README.md`)
- Run `python -m nerf.utils.data_utils --help` to see a description of all the available arguments.
- Use `python -m nerf.utils.data_utils` with the desired arguments.

### 2.4 Pose Info Format
The information regarding poses, bounds and intrinsics should be organized in a CSV file in a specific format. This format is referred to as the "**Pose Info Format**".

The CSV file should contain 6 columns. The columns are `image_name`, `camera_model`, `camera_params`, `pose`, `near` and `far`. Each row of the CSV file contains information regarding poses, bounds and intrinsics corresponding to the image mentioned in the same row.

A description of the data that should be present in each row for each column is given below:
1. `image_name`: 
	
	- Each row belonging to this column should contain a string denoting the name of the image.
2. `camera_model`: 
	- Each row belonging to this column should contain a string denoting a compatible camera model corresponding to the image. 
	- This codebase supports a subset of the camera models which are defined in [colmap](https://github.com/colmap/colmap). The exact camera models which are supported in this codebase are: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` and `FULL_OPENCV`.
	- The camera model, along with its corresponding camera params are used to construct the intrinsic matrix.
	- For more information regarding the camera models, please refer to the following file in the [colmap repository](https://github.com/colmap/colmap/blob/8bfff64843aea6c648ed7dfa5c28dd5b9d766b3b/src/base/camera_models.h).
3. `camera_params`:
	- Each row belonging to this column should contain a string denoting the camera parameters for the given camera model.
	- The first character of this string should be `[` and the last character of this string should be `]`. The characters in between the first and last characters of this string should contain comma separated values denoting the parameters for the given model. Inside the codebase, this string is parsed into a python list using `yaml.safe_load`.
		- For the exact values that are to be provided for a given camera model, please refer to this [file](https://github.com/colmap/colmap/blob/master/src/base/camera_models.h) in the colmap repository.
	- For example, if the corresponding camera model is `SIMPLE_PINHOLE`, then according to this [file](https://github.com/colmap/colmap/blob/master/src/base/camera_models.h) in the colmap repository, the parameters of the given model are `f`, `cx` and `cy`. Suppose we know that `f` is `500.0`, `cx` is `250.0` and `cy` is `200.0`, then the string that contains the camera parameters in our desired format is `[500.0, 250.0, 200.0]`
4. `pose`:
	
	- Each row belonging to this column should contain a string denoting the **camera to world transformation matrix** for the camera corresponding to the image. The camera to world transformation matrix must be in the **Classic CV format** (please refer to [coordinate_systems.md](docs/coordinate_systems.md) for information about the Classic CV format).
	
	- The first character of this string should be `[` and the last character of this string should be `]`. The characters in between the first and last characters of this string should contain comma separated values denoting the "**flattened pose matrix**". Inside the codebase, this string is parsed into a python list using `yaml.safe_load`.
	
	- A pose matrix is defined as the array `RT` of shape `(3, 4)` where `RT[:3, :3]` is a rotation matrix and `RT[:3, 3]` is a translation vector. On flattening `RT`, we would get a vector of shape `(12,)`. The **flattened pose matrix** are just these 12 values.
	
	  - The below example shows a NumPy array `RT` and its flattened version: 
	
	    ```python
	    >>> RT
	    array([[ 1. ,  0. ,  0. , -1.2],
	           [ 0. ,  1. ,  0. ,  3.4],
	           [ 0. ,  0. ,  1. , -7.2]])
	    >>> 
	    >>> RT.flatten()
	    array([ 1. ,  0. ,  0. , -1.2,  0. ,  1. ,  0. ,  3.4,  0. ,  0. ,  1. ,
	           -7.2])
	    ```
	
	  - For the above example, the string containing the flattened pose matrix in our desired format is `[ 1. ,  0. , 0. , -1.2,  0. ,  1. ,  0. ,  3.4,  0. ,  0. ,  1. , -7.2]`
5. `near`:
	
	-  Each row belonging to this column should contain the near bound for the camera corresponding to the image.
6. `far`:
	
	-  Each row belonging to this column should contain the far bound for the camera corresponding to the image.

An example of a table with two rows created in the "pose info format" is shown below:

| image_name | camera_model  | camera_params                                            | pose                                                         | near             | far              |
| ---------- | ------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ---------------- | ---------------- |
| 00180.png  | SIMPLE_RADIAL | [1240.1588277124777, 360.0, 360.0, 0.016340558510333107] | [0.9076337381480228, -0.2893411578144558, 0.3041096706289297, -1.4419341264172012, 0.418925793913631, 0.5786462489462741, -0.6997640300660218, 3.1480392638354595, 0.026498614478495372, 0.7625288276353862, 0.6464113322457485, 1.0811465031467153] | 3.57724263045598 | 9.21046974809568 |
| 00170.png  | SIMPLE_RADIAL | [1240.1588277124777, 360.0, 360.0, 0.016340558510333107] | [-0.20348249694604895, 0.8538044288490224, -0.47917937217121037, 3.177805065995965, -0.9690540429643606, -0.10577373141548839, 0.22303851585964013, -1.1015996869291331, 0.13974668243299365, 0.509735142029896, 0.8489057366567465, -0.10465786764184885] | 4.04273156716178 | 6.3928444260329  |

