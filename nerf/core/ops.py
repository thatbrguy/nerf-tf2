import numpy as np
import tensorflow as tf

from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback

def psnr_metric(y_true, y_pred):
    """
    Creating a metric function instead of a metric class.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = (-10.) * (tf.math.log(mse)/tf.math.log(10.))

    return psnr

class PSNRMetric(Metric):
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
