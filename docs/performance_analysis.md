TODO: intro

## Replication

As mentioned in `README.md`, determinism is not guaranteed. But, it is highly likely that a "close enough" replication of the evaluation performance mentioned above should be possible.

If the user is interested in replicating the evaluation experiment, they are follow the below steps:
1. Follow the instructions in section 5.1 of `README.md` to finish the common steps. 
	- As part of the common steps, the user would be instructed to prepare the data. Here, the user should setup the lego blender dataset.
2. Download the folder `predtrained_lego` (which is inside the `blender_datasets` folder) from this [link](https://drive.google.com/drive/folders/11jywsggHH5dJnOfnCeWJ5R4V95xd0S00?usp=sharing). This contains the pretrained weights, metadata and some logs. The same pretrained weights and metadata was used during the performance analysis.
3. For convenience, a config file has already been setup for this replication run (the file is `./nerf/params/config_lego.yaml`). 
	- To use the config file **as is**, the dataset folder (named `lego`), the downloaded `pretrained_lego` folder and this repository (named `nerf-tf2`) must lie in the same directory.
	- For example, let us assume the directory containing all three folders has the path `/path/to/root_dir`. Hence, the paths to each of the three folders should be: 
		- Path to the dataset folder: `/path/to/root_dir/lego`
		- Path to the `pretrained_lego` folder: `/path/to/root_dir/pretrained_lego`
		- Path to this repository: `/path/to/root_dir/nerf_tf2`
	- If the user would like to place the folders elsewhere, then they should edit the configuration file `./nerf/params/config_lego.yaml` to account for the paths.
4. Run the evaluation script using `python -m nerf.main.eval --config "./nerf/params/config_lego.yaml"`.
	- Do note that the steps related to setting up the environment etc. were already completed in step 1.

Using a V100 GPU on Google Colab, it took approximately one and a half hours to finish the full evaluation run.