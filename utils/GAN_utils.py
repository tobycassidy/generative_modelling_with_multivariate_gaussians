import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Union


def build_generator(generator_config: Dict[str, Union[int, List[int], List[str]]]):
    """
    
    """
    generator_input = tf.keras.Input(shape=generator_config['latent_dim'], name='generator_input')
    
    x = tf.keras.layers.Dense(np.prod(generator_config['reshape']), name='shape_prod')(generator_input)
    x = tf.keras.layers.Reshape(generator_config['reshape'], name='reshape')(x)
    
    for i, [f, k, a, s] in enumerate(zip(generator_config['filters'],
                                         generator_config['kernel_sizes'],
                                         generator_config['activations'],
                                         generator_config['strides'])):
        if a != 'sigmoid':
            x = tf.keras.layers.Conv2DTranspose(
                filters=f,
                kernel_size=k,
                activation=a,
                strides=s,
                padding='same',
                name='generator_conv_' + str(i + 1)
            )(x)
            x = tf.keras.layers.BatchNormalization(name='batch_norm_generator_' + str(i + 1))(x)
            x = tf.keras.layers.LeakyReLU(name='leaky_relu_generator_' + str(i + 1))(x)
        else:
            x = tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=k,
                activation=a,
                strides=s,
                padding='same',
                name='generator_conv_' + str(i + 1)
            )(x)
        
    generator_output = x
    generator = tf.keras.models.Model(
        inputs=generator_input,
        outputs=generator_output,
        name='generator'
    )
    return generator
        
    
def build_discriminator(discriminator_config: Dict[str, Union[int, List[int], List[str]]]):
    """
    
    """
    discriminator_input = tf.keras.Input(shape=discriminator_config['input_shape'], name='discriminator_input')
    x = discriminator_input
    
    for i, [f, k, a, s] in enumerate(zip(discriminator_config['filters'],
                                         discriminator_config['kernel_sizes'],
                                         discriminator_config['activations'],
                                         discriminator_config['strides'])):
        
        x = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=k,
            activation=a,
            strides=s,
            padding='same',
            name='discriminator_conv_' + str(i + 1)
        )(x)
        x = tf.keras.layers.BatchNormalization(name='batch_norm_discriminator_' + str(i + 1))(x)
        x = tf.keras.layers.LeakyReLU(name='leaky_relu_discriminator_' + str(i + 1))(x)
            
    x = tf.keras.layers.Flatten(name='flatten_discriminator')(x)
    discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid', name='discriminator_output')(x)
    
    discriminator = tf.keras.models.Model(
        inputs=discriminator_input,
        outputs=discriminator_output,
        name='discriminator'
    )
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return discriminator
    
    
def GAN(generator: tf.keras.models.Model, discriminator: tf.keras.models.Model):
    """
    
    """
    discriminator.trainable = False
    
    GAN = tf.keras.Sequential([
        generator,
        discriminator
    ])
    GAN.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return GAN

def generate_real_data(data, n_samples):
    """
    """
    idxs = np.random.randint(0, data.shape[0], n_samples)
    X = data[idxs]
    y = np.ones((n_samples, 1)) - 0.1 # label smmoothing
    return X, y

def generate_fake_data(generator, generator_config, n_samples, images: bool = True, inverse_labels: bool = False):
    """
    """
    noise = np.random.normal(0, 1, size=(n_samples, generator_config['latent_dim']))
    if images:
        X = generator.predict(noise)
    else:
        X = noise
    if inverse_labels:
        y = np.ones((n_samples, 1))
    else:
        y = np.zeros((n_samples, 1))
    return X, y

def generate_real_and_fake_data(data, generator, generator_config, n_samples):
    """
    """
    half_n_samples = n_samples // 2
    X_real, y_real = generate_real_data(data, half_n_samples)
    X_fake, y_fake = generate_fake_data(generator, generator_config, half_n_samples)
    X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
    return X, y

    
    
    
    
    
    
        