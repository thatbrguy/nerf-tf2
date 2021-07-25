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
    def __init__(self, params, val_spec):
        """
        TODO: Docstring.
        """
        super().__init__()
        self.params = params
        self.val_spec = val_spec

    def log_images(self, pred_batches):
        """
        Given a list of predictions of RGB values of pixels, this 
        function combines them into images based on the information 
        provided by self.val_spec.
        """
        # Shape of pred_pixels --> (L, 3)
        pred_pixels = tf.concat(pred_batches, axis = 0)
        start = 0
        
        # Since each image can have different (H, W), we are handling 
        # one image at a time.
        for idx, (H, W) in enumerate(self.val_spec):
            num_pixels_current_img = H * W
            end = start + num_pixels_current_img

            # Extracting all pixels that are part of the current image.
            pixels_current_img = pred_pixels[start:end, :]
            img = tf.reshape(pixels_current_img, (H, W, 3))

            ## TODO: Use [img] with img shape being (H, W, 3) OR 
            ## use img with img shape being (1, H, W, 3) ?? Will 
            ## both work?
            ## TODO: Check behavior and setting of max_outputs
            tf.summary.image(f"val_img_{idx}", [img], max_outputs = 1)

            # Updating start so that the next iteration of the loop 
            # can access the next image.
            start = end

        # Resetting self.model.val_cache so that the next validation 
        # run can use it properly.
        self.model.val_cache = []

    def on_test_end(self, logs):
        """
        TODO: Docstring.
        """
        # No action is taken if val_cache has no elements present.
        if len(self.model.val_cache) > 0:
            imgs = self.log_images(self.model.val_cache)

class CustomModelSaver(Callback):
    """
    This custom callback is used to save the models after 
    a validation run.

    TODO:
        1.  Consider saving optimzier and other relavent states as well.
        2.  Add support for NeRFLite as well. Currently the code 
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

        self.root = self.params.model.save_dir
        if not os.path.exists(self.root):
            os.mkdir(self.root)

    def _save(self, epoch, val_psnr_score):
        """
        Saves the model weights
        """
        name = f"{epoch: 06d}_{val_psnr_score: .2f}"
        coarse_model_name = f"{name}_coarse.h5"
        fine_model_name = f"{name}_fine.h5"
        
        coarse_model_path = os.path.join(self.root, coarse_model_name)
        fine_model_path = os.path.join(self.root, fine_model_name)

        self.model.coarse_model.save_weights(filepath = coarse_model_path)
        self.model.fine_model.save_weights(filepath = fine_model_path)

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
            self._save(epoch, val_psnr_score)


def psnr_metric(y_true, y_pred):
    """
    Creating a metric function instead of a metric class.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))

    return psnr

class PSNRMetric(Metric):
    """
    NOTE: Do not use this!
    """
    def __init__(self, name = "psnr_metric", **kwargs):
        super().__init__(name = name, **kwargs)
        self.psnr = self.add_weight(name = "psnr", initializer = "zeros")

    def update_state(self, y_true, y_pred, sample_weight = None):
        """
        IMPORTANT: sample_weight has no effect. TODO: Elaborate.
        """
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))
        
        ## TODO: Check if using assign is the right thing to do.
        self.psnr.assign(psnr)

    def result(self):
        return self.psnr

    def reset_states(self):
        self.psnr.assign(0.0)
