"""Configurable model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from .layers import scale_1p2

def build_model(input_shape, target_size,
                conv_size=16, kernel_size=2, n_conv_layers=5,
                fc1_size=128, fc2_size=64,
                hidden_activation='LeakyReLU',
                pooling_type='MaxPool3D',
                dropout=0,
                use_conv_bias=True,
                kernel_init='glorot_uniform'):
    """Construct the CosmoFlow 3D CNN model"""

    if kernel_init == "he_uniform_mod":
        leakyrelu_a = 0.3
        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.0/(1+leakyrelu_a*leakyrelu_a), mode='fan_in', distribution='uniform')

    conv_args = dict(kernel_size=kernel_size, padding='same', use_bias=use_conv_bias, kernel_initializer=kernel_init)
    hidden_activation_args = [] if hidden_activation == "LeakyReLU" else [hidden_activation] #[getattr(tf.keras.activations, hidden_activation)]
    hidden_activation = getattr(layers, hidden_activation) if hidden_activation == "LeakyReLU" else layers.Activation

    if pooling_type:
        pooling_type = getattr(layers, pooling_type)
    else:
        conv_args['strides'] = (2, 2, 2)

    model = tf.keras.models.Sequential()

    # First convolutional layer
    model.add(layers.Conv3D(conv_size, input_shape=input_shape, **conv_args))
    model.add(hidden_activation(*hidden_activation_args))
    if pooling_type:
        model.add(pooling_type(pool_size=2))

    # Additional conv layers
    for i in range(1, n_conv_layers):
        # Double conv channels at every layer
        model.add(layers.Conv3D(conv_size*2**i, **conv_args))
        model.add(hidden_activation(*hidden_activation_args))
        if pooling_type:
            model.add(pooling_type(pool_size=2))
    model.add(layers.Flatten())

    # Fully-connected layers
    model.add(layers.Dense(fc1_size, kernel_initializer=kernel_init))
    model.add(hidden_activation(*hidden_activation_args))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(fc2_size, kernel_initializer=kernel_init))
    model.add(hidden_activation(*hidden_activation_args))
    model.add(layers.Dropout(dropout))

    # Output layers
    model.add(layers.Dense(target_size, kernel_initializer=kernel_init, activation='tanh'))
    model.add(layers.Lambda(scale_1p2, dtype='float32'))

    return model
