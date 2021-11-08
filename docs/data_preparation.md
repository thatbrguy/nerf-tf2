# Data Preparation

This codebase currently supports two types of datasets:
1. The synthetic blender datasets which are mentioned in the official implementation.
2. Custom datasets which represent a "360-degree inward-facing" scene.

The following section elaborates how to prepare the dataset for the about two types.

## 1. Blender Dataset
Setting up a blender dataset is very simple. Please follow the below instructions:

1. Download the desired blender dataset from the GitHub repository of the [official implementation](https://github.com/bmild/nerf#running-code).
2. Place your desired blender dataset somewhere on your local disk and make note of the path to your blender dataset.
	- For example, if you are interested in the `lego` blender dataset, and you have downloaded and placed the lego dataset at `/path/to/datasets`, then the path the `lego` dataset would be `/path/to/datasets/lego`
3. Set the parameters in `nerf/params/config.yaml` appropriately by following the instructions given above each parameter in the file.

## 2. Custom Dataset (360-Degree Inward-Facing)

For any custom 360-degree inward-facing dataset, we require the following items: images, poses, bounds and intrinsics. TODO: rewrite. Collectively, we will refer to the images, poses, bounds and intrinsics as "information about the scene".

The codebase requires the information about the scene to be structured in a specific format. We will refer to this format as the "360-degree inward-facing custom dataset format". The format is as follows:
1. All the images should be stored in a single directory.
2. The information regarding the poses, bounds and intrinsics should be organized in a CSV file in the format specified in section 2.1 (Pose Info Format).

In some cases, the user might only have a set of images and would have no information about poses, bounds and intrinsics. In that case, the user can consider using the colmap-utils repository to extract.

If the user already has all the information about the scene, then the user could just organize the information into the format required by the codebase.

## 2.1 Pose Info Format
The information regarding poses, bounds and intrinsics should be organized in a CSV file in a specific format. This format is referred to as the "pose info format".

| image_name | camera_model  | camera_params                                            | pose                                                         | near             | far              |
| ---------- | ------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ---------------- | ---------------- |
| 00180.png  | SIMPLE_RADIAL | [1240.1588277124777, 360.0, 360.0, 0.016340558510333107] | [0.9076337381480228, -0.2893411578144558, 0.3041096706289297, -1.4419341264172012, 0.418925793913631, 0.5786462489462741, -0.6997640300660218, 3.1480392638354595, 0.026498614478495372, 0.7625288276353862, 0.6464113322457485, 1.0811465031467153] | 3.57724263045598 | 9.21046974809568 |
| 00170.png  | SIMPLE_RADIAL | [1240.1588277124777, 360.0, 360.0, 0.016340558510333107] | [-0.20348249694604895, 0.8538044288490224, -0.47917937217121037, 3.177805065995965, -0.9690540429643606, -0.10577373141548839, 0.22303851585964013, -1.1015996869291331, 0.13974668243299365, 0.509735142029896, 0.8489057366567465, -0.10465786764184885] | 4.04273156716178 | 6.3928444260329  |

