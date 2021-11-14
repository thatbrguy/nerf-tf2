import os
import logging
import numpy as np
import tensorflow as tf

from copy import deepcopy
from collections import defaultdict
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback

# Setting up logger
logger = logging.getLogger(__name__)

class LogValImages(Callback):
    """
    This custom callback is used to convert the pixel wise RGB 
    predictions to images, and then to log the images to tensorboard.
    
    IMPORTANT:
        1.  The TensorBoard callback MUST also be used for this 
            callback to work.
        2.  This callback works ONLY IF eager mode is 
            enabled AND IF image logging is enabled.
    """
    def __init__(self, params, height, width, num_val_imgs):
        super().__init__()
        raise NotImplementedError((
            "Temporarily dropping support for LogValImages until the "
            "functionality is tested and verified again."
        ))

        self.params = params
        self.img_width = width
        self.img_height = height
        self.num_imgs = num_val_imgs

        self.log_dir = os.path.join(self.params.system.tensorboard_dir, "imgs")

    def log_images(self, epoch, pred_batches):
        """
        Given a list of predictions of RGB values of pixels, this 
        function combines them into images.
        """
        # Shape of pred_pixels: (L, 3)
        # L = N * H * W, where N is the number of images, H is the height 
        # of each image and W is the width of each image.
        pred_pixels = tf.concat(pred_batches, axis = 0)
        H, W = self.height, self.width
        start = 0

        writer = tf.summary.create_file_writer(self.log_dir)
        with writer.as_default():

            for idx in range(self.num_imgs):
                num_pixels_current_img = H * W
                end = start + num_pixels_current_img

                # Extracting all pixels that are part of the current image.
                pixels_current_img = pred_pixels[start:end, :]
                img = tf.reshape(pixels_current_img, (H, W, 3))

                ## TODO: Use [img] with img shape being (H, W, 3) OR 
                ## use img with img shape being (1, H, W, 3) ?? Will 
                ## both work?
                ## TODO: Check behavior and setting of max_outputs
                tf.summary.image(
                    f"val_img_{idx}", [img], 
                    max_outputs = 1, step = epoch
                )
                writer.flush()

                # Updating start so that the next iteration of the loop 
                # can access the next image.
                start = end

        # Resetting self.model.val_cache so that the next validation 
        # run can use it properly.
        self.model.val_cache = []

    def on_epoch_end(self, epoch, logs = None):
        """
        Function to be called at the end of each epoch.
        """
        # No action is taken if val_cache has no elements present.
        if len(self.model.val_cache) > 0:
            imgs = self.log_images(epoch, self.model.val_cache)

class CustomSaver(Callback):
    """
    This custom callback is used to save a few items after each 
    validation run is complete. The items that are saved after 
    each validation run has been completed are:

        1. The weights of the coarse and fine models 
        2. The weights of the optimizer 
        3. The collected logs
    """
    def __init__(self, params, save_best_only = False):
        super().__init__()
        self.params = params
        self.best_score = -1
        self.collected_logs = defaultdict(list)
        self.save_best_only = save_best_only

        self.save_opt_state = self.params.model.save.save_optimizer_state
        self.root = self.params.model.save.save_dir
        if not os.path.exists(self.root):
            os.mkdir(self.root)

    def _save_weights(self, path, variables):
        """
        Saves weights to a .npz file given a list of variables.
        """
        variable_names = [x.name for x in variables]
        variable_values = [x.numpy() for x in variables]
        items = {k:v for k, v in zip(variable_names, variable_values)}
        
        # Saving names as well to preserve order of the variables names.
        items["names"] = np.array(variable_names)
        np.savez(path, **items)

    def _save_logs(self, path):
        """
        Saves the collected logs to a .npz file.
        """
        to_save = deepcopy(self.collected_logs)
        np.savez(path, **to_save)

    def _save_everything(self, epoch, val_psnr_score):
        """
        Saves the model weights.
        """
        name = f"{epoch:06d}_{val_psnr_score:.2f}"
        coarse_model_name = f"{name}_coarse.h5"
        fine_model_name = f"{name}_fine.h5"
        logs_name = f"{name}_logs.npz"
        
        coarse_model_path = os.path.join(self.root, coarse_model_name)
        fine_model_path = os.path.join(self.root, fine_model_name)
        logs_path = os.path.join(self.root, logs_name)

        self.model.coarse_model.save_weights(filepath = coarse_model_path)
        self.model.fine_model.save_weights(filepath = fine_model_path)
        self._save_logs(logs_path)

        if self.save_opt_state:
            opt_save_path =  os.path.join(self.root, f"{name}_optimizer.npz")
            opt_variables = self.model.optimizer.variables()
            self._save_weights(opt_save_path, opt_variables)

    def on_epoch_end(self, epoch, logs):
        """
        Function to be called at the end of each epoch.
        """
        # Updating self.collected_logs
        self.collected_logs["train_epoch_idxs"].append(epoch)
        self.collected_logs["train_psnr_metric"].append(logs["psnr_metric"])
        
        try:
            val_psnr_score = logs["val_psnr_metric"]

        except KeyError:
            logger.debug(
                f"val_psnr_metric not found for epoch {epoch}. Skipping."
            )
            return

        self.collected_logs["val_epoch_idxs"].append(epoch)
        self.collected_logs["val_psnr_metric"].append(val_psnr_score)
        
        if self.save_best_only:
            if val_psnr_score > self.best_score:
                logger.debug(f"Saving model weights for epoch {epoch}.")
                self._save_everything(epoch, val_psnr_score)
                self.best_score = val_psnr_score
            else:
                logger.debug(
                    f"Not saving anything for {epoch} since "
                    "val_psnr_score <= self.best_score and "
                    "since self.save_best_only is True"
                )

        else:
            logger.debug(f"Saving model weights for epoch {epoch}.")
            self._save_everything(epoch, val_psnr_score)

class PSNRMetric(Metric):
    """
    Custom metric to computes PSNR.

    TODO: mention that this class is to be used with the model
    """
    def __init__(self, name = "psnr_metric", **kwargs):
        super().__init__(name = name, **kwargs)
        
        self.sq_error = self.add_weight(
            name = "metric_vars/sq_error", initializer = "zeros"
        )
        
        self.count = self.add_weight(
            name = "metric_vars/count", initializer = "zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight = None):
        """
        Updates the states.
    
        For information about the arguments of this function, refer to:
            https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics

        On each call of this function, self.sq_error and self.count is updated.
        Overall, self.sq_error keeps track of the sum squared error so far, and
        self.count keeps track of the number of elements for which the sum
        squared error has been computed so far.
        """
        sq_error = tf.reduce_sum(tf.square(y_true - y_pred))
        count = tf.cast(tf.shape(y_true)[0], tf.float32)
        
        self.sq_error.assign_add(sq_error)
        self.count.assign_add(count)

    def result(self):
        """
        Computes the PSNR and returns the result.
        """
        # Computing mean squared error
        mse = self.sq_error / self.count
        # Computing psnr
        psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))

        return psnr

    def reset_states(self):
        """
        Resets the states to their initial values.
        """
        self.sq_error.assign(0.0)
        self.count.assign(0.0)

def psnr_metric(y_true, y_pred):
    """
    A function to compute PSNR. The inputs must be TensorFlow tensors.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))

    return psnr

def psnr_metric_numpy(y_true, y_pred):
    """
    A function to compute PSNR. The inputs must be NumPy arrays.
    TODO: Check if consistent with psnr_metric function.
    """
    mse = np.mean(np.square(y_true - y_pred))
    psnr = (-10.) * (np.log(mse)/np.log(10.))

    return psnr
