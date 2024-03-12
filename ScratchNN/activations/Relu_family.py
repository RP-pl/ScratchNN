import numpy as np
import tensorflow as tf

import ScratchNN.activations


@tf.function
def relu(x):
    """Rectified Linear Unit activation function"""
    return tf.maximum(x, tf.constant(np.zeros(x.shape), dtype=x.dtype))


@tf.function
def elu(x):
    """Exponential Linear Unit activation function"""
    return tf.where(x > 0, x, tf.exp(x) - 1)


@tf.function
def selu(x):
    """Scaled Exponential Linear Unit activation function"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0, x, alpha * tf.exp(x) - alpha)


@tf.function
def softplus(x):
    """Softplus activation function"""
    return tf.math.log(1 + tf.exp(x))


@tf.function
def swish(x):
    """Swish activation function"""
    return x * ScratchNN.activations.sigmoid(x)


@tf.function
def leaky_relu(x, alpha=0.3):
    """Leaky ReLU activation function"""
    return tf.maximum(x, alpha * x)


@tf.function
def linear(x):
    """Linear activation function"""
    return x
