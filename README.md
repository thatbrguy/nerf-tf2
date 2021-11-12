# NeRF TF2

An unofficial implementation of NeRF in TensorFlow 2 for 360-degree inward-facing scenes *(forward-facing scenes are currently not supported -- will add support sometime late December 2021).*

## 1. Features and Highlights
- Contains a full implementation of the NeRF model for 360-degree inward-facing scenes.
- TODO: write about the NeRF class implementation
- TODO: finish this section

## 2. Important Notes
TODO

### 2.1. Major Differences
- The camera coordinate system format used in this implementation and the official implementation is the camera coordinate system format.
	- This codebase mostly uses the Classic CV coordinate system format, whereas the official implementation mostly uses the OpenGL coordinate system format. For information about the various coordinate system formats mentioned in this codebase, please refer to [coordinate_systems.md](docs/coordinate_systems.md).
	- The above choice was made since I thought that data preparation for custom real life data would be simpler this way. Moreover I believed that the codebase would be a bit easier to understand if the Classic CV format is used.
- Inverse transform sampling has been implemented from scratch in a different way. A different strategy to handle edge cases is employed.
- TODO: finish this section

### 2.2 Currently Unsupported Features
- Forward-facing scenes are currently not supported. Support is planned to be added sometime late December 2021.

## 3. Performance Analysis
TODO

## 4. Setup and Data Preparation
This section instructions on how to setup the codebase for usage, and also on how to prepare the data for usage with this codebase.

### 4.1. Setup
Follow the below instructions to setup the codebase for usage:
1. Launch a terminal. 
2. Create a virtual environment using your preferred method. Activate the virtual environment.
3. Git clone this repository and change your directory to the location of this repository in your local filesystem.
	- For example, if you git cloned this repository to `/path/to/nerf-tf2`, please run the command `cd /path/to/nerf-tf2` in your terminal.
4. Install the dependencies using the the following command: `pip install -r requirements.txt`

### 4.2. Data Preparation
This codebase currently supports two types of datasets:
1. The synthetic blender datasets which are mentioned in the official implementation.
2. Custom datasets which represent an "360 degree inward-facing" scene.

For instructions on preparing your dataset, please refer to the document [data_preparation.md](docs/data_preparation.md).

## 5. Usage
This section contains instructions on how to use various components of this codebase.

### 5.1. Common Steps
1. Ensure that the dependencies have been installed (section 4.1) and the data has been prepared (section 4.2)
2. Launch a terminal and activate the virtual environment where you had previously installed the dependencies.
3. Change your directory to the location of this repository in your local filesystem.
	- For example, if you git cloned this repository to `/path/to/nerf-tf2`, please run the command `cd /path/to/nerf-tf2` in your terminal.

### 5.2. Training
Follow the below instructions to launch a training run:

1. Follow the procedure mentioned in the common steps section (section 5.1).
2. Modify the parameters in the configuration file `nerf/params/config.yaml` as per your requirements.
3. Run the training script by running the command `python -m nerf.main.train`

### 5.3. Evaluation
Follow the below instructions to launch an evaluation run:

1. Follow the procedure mentioned in the common steps section (section 5.1).
2. Modify the parameters in the configuration file `nerf/params/config.yaml` as per your requirements.
3. Run the training script by running the command `python -m nerf.main.eval`

### 5.4. Rendering
Follow the below instructions to launch a rendering run:

1. Follow the procedure mentioned in the common steps section (section 5.1).
2. Modify the parameters in the configuration file `nerf/params/config.yaml` as per your requirements.
3. Run the training script by running the command `python -m nerf.main.render`

### 5.5. Visualization
Follow the below instructions to visualize various elements of the scene:

1. Follow the procedure mentioned in the common steps section (section 5.1).
2. Modify the parameters in the configuration file `nerf/params/config.yaml` as per your requirements.
3. Run the training script by running the command `python -m nerf.main.viz_scene`

## 6. Acknowledgement
TODO

## 7. Citation
TODO
