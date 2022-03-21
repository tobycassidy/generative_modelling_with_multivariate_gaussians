import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Union



class KLDivergenceLayer(tf.keras.layers.Layer):
    """
    Identity transform layer,
    only purpose is to add the kl_loss for training.
    """
    def __init__(self, *args, **kwargs):
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        
    def call(self, distribution_parameters):
        z_mean, z_log_var = distribution_parameters
        
        kl_batch = -0.5 * tf.math.reduce_sum(
            z_log_var - tf.exp(z_log_var) - tf.square(z_mean) + 1,
            axis=-1
        )
        
        self.add_loss(tf.math.reduce_mean(kl_batch), inputs=distribution_parameters)
        
        return distribution_parameters
    
    
class Sampling(tf.keras.layers.Layer):
    """
    Sampling layer,
    using the reparametrization technique to ensure randomness independence and thus allowing the passing of gradients.
    """
    def __init__(self, *args, **kwargs):
        super(Sampling, self).__init__(*args, **kwargs)
        
    def call(self, distribution_parameters):
        z_mean, z_log_var = distribution_parameters
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), mean=0.0, stddev=0.1)
        
        return z_mean + epsilon * tf.exp(0.5 * z_log_var)
    
    
class VAE(tf.keras.models.Model):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

def build_variational_encoder(encoder_config: Dict[str, Union[int, List[int], List[str]]]):
    """
    
    """
    encoder_input = tf.keras.Input(shape=(28, 28, 1), name='mnist_inputs')
    x = encoder_input
    
    for i, [f, k, a, s] in enumerate(zip(encoder_config['filters'], 
                                         encoder_config['kernel_sizes'], 
                                         encoder_config['activations'],
                                         encoder_config['strides'])):
        
        x = tf.keras.layers.Conv2D(
            filters=f, 
            kernel_size=k,
            activation=a,
            strides=s,
            padding='same',
            name='encoder_conv_' + str(i + 1)
        )(x)
        x = tf.keras.layers.BatchNormalization(name='batch_norm_' + str(i + 1))(x)
        x = tf.keras.layers.LeakyReLU(name='leaky_relu_encoder_' + str(i + 1))(x)
        
    x = tf.keras.layers.Flatten(name='flatten_encoder')(x)
    x = tf.keras.layers.Dense(encoder_config['latent_dim'], name='encoder_output')(x)
    
    
    z_mean = tf.keras.layers.Dense(encoder_config['latent_dim'], name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(encoder_config['latent_dim'], name='z_log_var')(x)
    #z_mean, z_log_var = KLDivergenceLayer(name='kl_divergence_identity')([z_mean, z_log_var])
    z = Sampling(name='sampling')([z_mean, z_log_var])
    encoder_output = z
    
    encoder = tf.keras.models.Model(
        inputs=encoder_input,
        outputs=[z_mean, z_log_var, z],
        name='encoder'
    )
    
    
    return encoder_input, encoder_output, encoder, z_mean, z_log_var


def build_variational_decoder(decoder_config : Dict[str, Union[int, List[int], List[str]]]):
    """
    
    """
    decoder_input = tf.keras.Input(shape=decoder_config['latent_dim'], name='decoder_input')
    
    x = tf.keras.layers.Dense(np.prod(decoder_config['reshape']), name='shape_prod')(decoder_input)
    x = tf.keras.layers.Reshape(decoder_config['reshape'])(x)
    
    for i, [f, k, a, s] in enumerate(zip(decoder_config['filters'], 
                                         decoder_config['kernel_sizes'], 
                                         decoder_config['activations'],
                                         decoder_config['strides'])):
        
        x = tf.keras.layers.Conv2DTranspose(
            filters=f, 
            kernel_size=k,
            activation=a,
            strides=s,
            padding='same',
            name='decoder_conv_' + str(i + 1)
        )(x)
        
            
    decoder_output = x
    decoder = tf.keras.models.Model(
        inputs=decoder_input,
        outputs=decoder_output,
        name='decoder'
    )
    
    
    return decoder_input, decoder_output, decoder