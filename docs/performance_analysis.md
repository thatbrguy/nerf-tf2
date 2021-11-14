# Performance Analysis

## 1. Lego Dataset
- On the lego test set, using the script `evaluate.py` an **Mean PSNR** of **33.0369** was obtained. Using a single V100 GPU on Google Colab, the evaluation took approximately **26.75 seconds per image** (as per the tqdm logs -- also this time includes the time it takes to save each image to disk).
- **However**, the user must be cautious while interpreting this result for many reasons:
	- There are multiple ways of calculating the Mean PSNR metric (for example, calculating the mean PSNR per image and then averaging across images, versus calculating the mean PSNR per batch of pixels and then averaging across all batches etc.).
	- During the training process, the validation PSNR obtained for the saved weights used for this analysis was around 27.67. However, this also should be taken with caution, since only 3 images were used for validation and also because the way PSNR is calculated in `evaluate.py` may be different from the way PSNR was calculated during validation.
	- Also, some configurations were different when compared to the original implementation (for example, here a batch size of 4096 was used).
- In any case, the above information is provided to the user so that they can make a more careful interpretation of the results.
- Further analysis will be provided in the near future.
- In the section 1.1 of this document, instructions are provided so that the user can attempt to replicate the evaluation experiment for the lego test set.

## 1.1 Replication

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

## 2. Custom Dataset
- The performance on the custom dataset will elaborated on in the near future.
