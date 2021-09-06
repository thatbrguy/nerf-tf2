import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback

# Setting up logger
logger = logging.getLogger(__name__)

class LogValImages(Callback):
    """
    This custom callback is used to convert the pixel wise RGB 
    predictions to images, and then log the images to tensorboard.

    The TensorBoard Callback MUST also be used for this callback to work!
    """
    def __init__(self, params, height, width, num_val_imgs):
        """
        TODO: Docstring.
        """
        super().__init__()
        self.params = params
        self.img_width = width
        self.img_height = height
        self.num_imgs = num_val_imgs

        self.log_dir = os.path.join(self.params.system.tensorboard_dir, "imgs")

    def log_images(self, epoch, pred_batches):
        """
        Given a list of predictions of RGB values of pixels, this 
        function combines them into images based on the information 
        provided by self.val_spec.
        """
        # Shape of pred_pixels --> (L, 3)
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
        TODO: Docstring.
        """
        # No action is taken if val_cache has no elements present.
        if len(self.model.val_cache) > 0:
            imgs = self.log_images(epoch, self.model.val_cache)

class CustomModelSaver(Callback):
    """
    This custom callback is used to save the models after 
    a validation run.

    TODO:
        1.  Add support for NeRFLite as well. Currently the code 
            assumes that the fine model is available and will try to 
            save the weights of the fine model as well.
    """
    def __init__(self, params, save_best_only = False):
        """
        TODO: Docstring.
        """
        super().__init__()
        self.params = params
        self.best_score = -1
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

    def _save_everything(self, epoch, val_psnr_score):
        """
        Saves the model weights
        """
        ## TODO: Print and see how the names look.
        name = f"{epoch:06d}_{val_psnr_score:.2f}"
        coarse_model_name = f"{name}_coarse.h5"
        fine_model_name = f"{name}_fine.h5"
        
        coarse_model_path = os.path.join(self.root, coarse_model_name)
        fine_model_path = os.path.join(self.root, fine_model_name)

        self.model.coarse_model.save_weights(filepath = coarse_model_path)
        self.model.fine_model.save_weights(filepath = fine_model_path)

        if self.save_opt_state:
            opt_save_path =  os.path.join(self.root, f"{name}_optimizer.npz")
            opt_variables = self.model.optimizer.variables()
            self._save_weights(opt_save_path, opt_variables)

    def on_epoch_end(self, epoch, logs = None):
        """
        TODO: Docstring.
        """
        try:
            val_psnr_score = logs["val_psnr_metric"]

        except KeyError:
            logger.debug(
                f"val_psnr_metric not found for epoch {epoch}. Skipping."
            )
            return
        
        if self.save_best_only:
            if val_psnr_score > self.best_score:
                logger.debug(f"Saving model weights for epoch {epoch}.")
                self._save(epoch, val_psnr_score)
                self.best_score = val_psnr_score

        else:
            logger.debug(f"Saving model weights for epoch {epoch}.")
            self._save_everything(epoch, val_psnr_score)

class PSNRMetric(Metric):
    """
    Computes PSNR

    TODO: Elaborate functionality for all functions
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
        IMPORTANT: sample_weight has no effect.

        TODO: Elaborate functionality.
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
        self.sq_error.assign(0.0)
        self.count.assign(0.0)

def psnr_metric(y_true, y_pred):
    """
    Creating a metric function instead of a metric class.

    NOTE: Do note use this! Only kept here for reference purposes.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))

    return psnr
