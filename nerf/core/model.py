import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras.layers import Input, Concatenate

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
        self.val_cache = []

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.coarse_model = get_nerf_model(model_name = "coarse")
        self.fine_model = get_nerf_model(model_name = "fine")

    def call(self, inputs):
        """
        TODO: Docstring.
        """
        output = self.forward(*inputs)
        return output

    def forward(self, rays_o, rays_d, near, far):
        """
        Performs a forward pass.
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
            t_vals = data_CM["t_vals"]
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
            t_vals = data_FM["t_vals"]
        )

        return post_proc_CM, post_proc_FM

    def train_step(self, data):
        """
        TODO: Docstring!

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
        TODO: Docstring!

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
        # after one validation run is complete.
        ## NOTE: The val images logger is not working as 
        ## expected and hence is not currently being used. 
        ## Will think about an alternative way to log images later.
        # self.val_cache.append(post_proc_FM["pred_rgb"])

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        pass

    def render(self, data):
        """
        Renders an image.

        Requires eager mode!
        """
        pixels = []
        (intrinsic, c2w, H, W) = data
        
        # Here, L denotes the number of pixels in this image.
        L = H * W
        far_all = tf.ones(shape = (L, 1), dtype = tf.float32)
        near_all = tf.zeros(shape = (L, 1), dtype = tf.float32)
        rays_o_all, rays_d_all = ray_utils.get_rays_tf(H, W, intrinsic, c2w)

        data = (rays_o_all, rays_d_all, near_all, far_all)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(
            batch_size = self.params.data.batch_size, 
            drop_remainder = False,
        )

        for batch in dataset:
            (rays_o, rays_d, near, far) = batch
            
            # Performing a forward pass.
            post_proc_CM, post_proc_FM = self.forward(
                rays_o = rays_o, rays_d = rays_d, 
                near = near, far = far,
            )
            batch_pixels = post_proc_FM["pred_rgb"].numpy()
            pixels.append(batch_pixels)

        pixels = np.concatenate(pixels, axis = 0)
        img = np.reshape(pixels, (H, W, 3))
        img = np.clip(img * 255.0, 0, 255)
        img = img.astype(np.uint8)
        
        return img

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

class NeRFLite(Model):
    """
    Model to define custom fit, evaluate and predict operations.

    This class implements only the coarse model.

    Legend:
        CM  : Coarse Model
    """
    def __init__(self, params):
        
        super().__init__()
        self.params = params

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.coarse_model = get_nerf_model(model_name = "coarse")

    def train_step(self, data):
        """
        TODO: Docstring
        """
        (rays_o, rays_d, near, far), (rgb,) = data

        # Getting data ready for the coarse model.
        data_CM = ray_utils.create_input_batch_coarse_model(
            params = self.params, rays_o = rays_o, 
            rays_d = rays_d, near = near, far = far
        )

        # Performing a forward pass through the coarse model.
        with tf.GradientTape() as tape:
            rgb_CM, sigma_CM = self.coarse_model(
                inputs = (data_CM["xyz_inputs"], data_CM["dir_inputs"])
            )
            
            post_proc_CM = ray_utils.post_process_model_output(
                sample_rgb = rgb_CM, sigma = sigma_CM, 
                t_vals = data_CM["t_vals"]
            )

            # Computing coarse loss.
            coarse_loss = self.mse_loss(
                y_true = rgb, 
                y_pred = post_proc_CM["pred_rgb"]
            )

        gradients = tape.gradient(coarse_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ## TODO: Make note somewhere. Metric is computed for the coarse model 
        ## output for the class NeRFLite.
        self.compiled_metrics.update_state(
            y_true = rgb, 
            y_pred = post_proc_CM["pred_rgb"]
        )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        pass

    def predict_step(self, data):
        pass

class PositionalEncoder(Layer):
    """
    TODO: Docstring and verify code!
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

    def call(self, x):
        """
        TODO: Docstring
        """
        expanded = tf.expand_dims(x, axis = -1) * self.multipliers
        
        cos_ = tf.math.cos(expanded)
        sin_ = tf.math.sin(expanded)

        intermediate = tf.stack([sin_, cos_], axis = -1)

        ## TODO: Check if using Flatten is correct.
        # output = tf.reshape(intermediate, (x.shape[0], -1))
        output = Flatten()(intermediate)

        return output

def get_nerf_model(model_name, num_units = 256):
    """
    Creates and returns a NeRF model.

    ## TODO: Verify consistency, indices, check for one off errors etc.
    ## Add comments to explain stuff!

    ## NOTE: relu is added to sigma here itself.
    """
    assert model_name in ["coarse", "fine"]
    
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

if __name__ == '__main__':

    from nerf.utils.params_utils import load_params

    path = "./nerf/params/config.yaml"
    params = load_params(path)

    coarse_model = get_nerf_model(model_name = "coarse")
    fine_model = get_nerf_model(model_name = "fine")

    nerf = NeRF(params)
    psnr = ops.PSNRMetric()
    nerf.compile(optimizer = 'adam', metrics = [psnr])
    
    nerf_lite = NeRFLite(params)
