import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras.layers import Input, Concatenate

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from nerf.core import ops
from nerf.utils import ray_utils, pose_utils

class NeRF(Model):
    """
    Model to define custom fit, evaluate and predict operations.

    This class implements both coarse and fine models.
    """
    def __init__(self, params):

        super().__init__()
        self.params = params
        self.white_bg = self.params.system.white_bg
        
        self.val_cache = []
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.coarse_model = get_coarse_or_fine_model(model_name = "coarse")
        self.fine_model = get_coarse_or_fine_model(model_name = "fine")

    def call(self, inputs):
        """
        TODO: Docstring.
        """
        output = self.forward(*inputs[0])
        return output

    def forward(self, rays_o, rays_d, near, far):
        """
        Performs a forward pass.

        Args:
            rays_o          :
            rays_d          :
            near            :
            far             :

        Returns:
            post_proc_CM    :
            post_proc_FM    :
        """
        # Getting data ready for the coarse model.
        ## TODO: Verify if it is ok for the below line 
        ## to be within GradientTape.
        data_CM = ray_utils.create_input_batch_coarse_model(
            params = self.params, rays_o = rays_o, 
            rays_d = rays_d, near = near, far = far
        )
        
        # Performing a forward pass through the coarse model.
        rgb_CM, sigma_CM = self.coarse_model(
            inputs = (data_CM["xyz_inputs"], data_CM["dir_inputs"])
        )
        
        # Postprocessing coarse model output.            
        post_proc_CM = ray_utils.post_process_model_output(
            sample_rgb = rgb_CM, sigma = sigma_CM, 
            t_vals = data_CM["t_vals"], 
            white_bg = self.white_bg,
        )

        # Getting data ready for the fine model.
        data_FM = ray_utils.create_input_batch_fine_model(
            params = self.params, rays_o = rays_o, rays_d = rays_d, 
            bin_weights = post_proc_CM["weights"], 
            t_vals_coarse = data_CM["t_vals"],
            bin_data = data_CM["bin_data"],
        )

        # Performing a forward pass through the fine model.
        rgb_FM, sigma_FM = self.fine_model(
            inputs = (data_FM["xyz_inputs"], data_FM["dir_inputs"])
        )

        # Postprocessing fine model output.
        post_proc_FM = ray_utils.post_process_model_output(
            sample_rgb = rgb_FM, sigma = sigma_FM, 
            t_vals = data_FM["t_vals"],
            white_bg = self.white_bg,
        )

        return post_proc_CM, post_proc_FM

    def train_step(self, data):
        """
        Custom train step.

        Legend:
            CM  : Coarse Model
            FM  : Fine Model
        """
        (rays_o, rays_d, near, far), (rgb,) = data
        
        with tf.GradientTape() as tape:
            
            # Performing a forward pass.
            post_proc_CM, post_proc_FM = self.forward(
                rays_o = rays_o, rays_d = rays_d, 
                near = near, far = far,
            )

            # Computing coarse loss.
            coarse_loss = self.mse_loss(
                y_true = rgb, 
                y_pred = post_proc_CM["pred_rgb"]
            )

            # Computing fine loss
            fine_loss = self.mse_loss(
                y_true = rgb, 
                y_pred = post_proc_FM["pred_rgb"]
            )

            total_loss = coarse_loss + fine_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ## TODO: Make note somewhere. Metric is computed only for the 
        ## fine model output for the class NeRF. Change if required.
        self.compiled_metrics.update_state(
            y_true = rgb, 
            y_pred = post_proc_FM["pred_rgb"]
        )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Custom test step.

        Legend:
            CM  : Coarse Model
            FM  : Fine Model
        """
        (rays_o, rays_d, near, far), (rgb,) = data
        
        # Performing a forward pass.
        post_proc_CM, post_proc_FM = self.forward(
            rays_o = rays_o, rays_d = rays_d, 
            near = near, far = far,
        )

        ## TODO: Make note somewhere. Metric is computed only for the 
        ## fine model output for the class NeRF. Change if required.
        self.compiled_metrics.update_state(
            y_true = rgb, 
            y_pred = post_proc_FM["pred_rgb"]
        )

        # Saving predictions to val_cache so that LogValImages can 
        # use val_cache to log validation images to TensorBoard 
        # after one validation run is complete. Only works 
        # with eager mode.
        if self.params.system.run_eagerly and self.params.system.log_images:
            self.val_cache.append(post_proc_FM["pred_rgb"].numpy())

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """
        Custom predict step.

        TODO: Check correctness
        """
        return self(data)

    def set_everything(self):
        """
        This function can be called AFTER model.compile has been 
        called to set the weights of coarse model, fine model and 
        the optimizer.
        """
        load_params = self.params.model.load
        root = load_params.load_dir

        if not load_params.skip_optimizer:
            # Performing an optimizer step so that self.optimizer.variables() 
            # gets populated with variables. We use zero gradients for all the 
            # trainable variables. I think this step is fine since after 
            # self.optimizer.variables() gets populated, we will set 
            # all the variables (of the coarse model, fine model and the 
            # optimizer) to the saved values.
            trainable_vars = self.trainable_variables
            gradients = [tf.zeros_like(x) for x in trainable_vars]
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            optimizer_variables = self.optimizer.variables()
            current_var_names = np.array([x.name for x in optimizer_variables])

            optimizer_weights_path = os.path.join(
                root, f"{load_params.load_tag}_optimizer.npz"
            )

            optimizer_data = np.load(optimizer_weights_path)
            saved_var_names = optimizer_data["names"]

            assert np.all(current_var_names == saved_var_names), (
                "The optimizer state cannot be loaded since the "
                "current variable names and saved variable names "
                "are different."
            )

            optimizer_weights = [optimizer_data[str(k)] for k in saved_var_names]
            self.optimizer.set_weights(optimizer_weights)

        coarse_model_weights_path = os.path.join(
                root, f"{load_params.load_tag}_coarse.h5"
        )
        fine_model_weights_path = os.path.join(
                root, f"{load_params.load_tag}_fine.h5"
        )

        # Now, we load the weights of the coarse model and the fine model.
        self.coarse_model.load_weights(coarse_model_weights_path)
        self.fine_model.load_weights(fine_model_weights_path)

class PositionalEncoder(Layer):
    """
    TODO: 

    1.  Add docstring and verify code!
    2.  Consider adding a parameter so that the user can 
        multiply ((2 ** exponents) * np.pi) with that parameter. 
        This can be used to control the periodicity of the sin 
        and cos functions.
    3.  Allow the user to decide if they want to concatenate 
        the input with the positional encoded values. Curently 
        it is concatenated.
    """
    def __init__(self, L, name = None):
        """
        TODO: Docstring
        """
        super().__init__(trainable = False, name = name)

        self.L = L
        exponents = tf.range(self.L, dtype = tf.float32)
        self.multipliers = (2 ** exponents) * np.pi
        self.multipliers =  tf.reshape(self.multipliers, (1, 1, -1))

    def build(self, input_shape):
        """
        TODO: Docstring
        """
        self.last_dim = input_shape[1] * self.L * 2

    def call(self, x):
        """
        TODO: Docstring
        """
        expanded = tf.expand_dims(x, axis = -1) * self.multipliers
        
        cos_ = tf.math.cos(expanded)
        sin_ = tf.math.sin(expanded)

        intermediate = tf.stack([sin_, cos_], axis = -1)

        ## TODO: Check if Flatten can be used.
        # output = Flatten()(intermediate)
        sincos = tf.reshape(intermediate, (-1, self.last_dim))
        output = tf.concat([x, sincos], axis = -1)

        return output

def get_coarse_or_fine_model(model_name, num_units = 256):
    """
    Creates and returns either the coarse model or the fine model.

    ## TODO: Verify consistency, indices, check for one off errors etc.
    ## Add comments to explain stuff!

    ## NOTE: relu is added to sigma here itself.
    """
    assert model_name in ("coarse", "fine")
    
    xyz = Input(shape = (3,), name = f"{model_name}/xyz")
    rays_d = Input(shape = (3,), name = f"{model_name}/rays_d")

    name = f"{model_name}/enc_xyz"
    enc_xyz = PositionalEncoder(L = 10, name = name)(xyz)

    name = f"{model_name}/enc_rays_d"
    enc_rays_d = PositionalEncoder(L = 4, name = name)(rays_d)

    value = enc_xyz

    for i in range(8):
        name = f"{model_name}/dense_{i}"
        value = Dense(num_units, name = name, activation = "relu")(value)

        if i == 4:
            name = f"{model_name}/concat_1"
            value = Concatenate(name = name)([value, enc_xyz])

    name =  f"{model_name}/sigma"
    sigma = Dense(1, activation = "relu", name = name)(value)

    name =  f"{model_name}/dense_8"
    bottleneck = Dense(num_units, activation = None, name = name)(value)
    
    name = f"{model_name}/concat_2"
    value = Concatenate(name = name)([bottleneck, enc_rays_d])
    
    name = f"{model_name}/dense_9"
    value = Dense(num_units // 2, name = name, activation = "relu")(value)

    name = f"{model_name}/rgb"
    rgb = Dense(3, activation = "sigmoid", name = name)(value)

    inputs = [xyz, rays_d]
    outputs = [rgb, sigma]

    model = Model(inputs = inputs, outputs = outputs)

    return model

def setup_model(params):
    """
    Sets up the NeRF model
    """
    # Setting up the optimizer and LR scheduler.
    lr_schedule = ExponentialDecay(
        initial_learning_rate = 5e-4,
        decay_steps = 500000,
        decay_rate = 0.1,
    )
    optimizer = Adam(learning_rate = lr_schedule)
    psnr_metric = ops.PSNRMetric()

    # Instantiating the model.
    nerf = NeRF(params = params)
    nerf.compile(
        optimizer = optimizer, 
        metrics = [psnr_metric], 
        run_eagerly = params.system.run_eagerly,
    )

    if params.model.load.set_weights:
        nerf.set_everything()

    return nerf

def setup_model_and_callbacks(params, num_imgs, img_HW):
    """
    Sets up the NeRF model and the list of callbacks.
    """
    # Setting up callbacks.
    saver = ops.CustomSaver(params = params, save_best_only = False)
    tensorboard = TensorBoard(
        log_dir = params.system.tensorboard_dir, 
        update_freq = "epoch"
    )
    callbacks = [saver, tensorboard]
    
    # Can use ops.LogValImages only if eager mode is enabled.
    if params.system.run_eagerly and params.system.log_images:
        val_imgs_logger = ops.LogValImages(
            params = params, height = img_HW[0], 
            width = img_HW[1], num_val_imgs = num_imgs["val"]
        )
        callbacks += [val_imgs_logger]

    # Setting up the NeRF model.
    nerf = setup_model(params)

    return nerf, callbacks

if __name__ == '__main__':

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    coarse_model = get_coarse_or_fine_model(model_name = "coarse")
    fine_model = get_coarse_or_fine_model(model_name = "fine")

    nerf = NeRF(params)
    psnr = ops.PSNRMetric()
    nerf.compile(optimizer = 'adam', metrics = [psnr])
