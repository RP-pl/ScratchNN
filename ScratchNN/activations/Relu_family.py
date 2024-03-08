import numpy as np
import tensorflow as tf


@tf.function
def relu(x):
    return tf.maximum(x, tf.constant(np.zeros(x.shape), dtype=x.dtype))


@tf.function
def elu(x):
    return tf.where(x > 0, x, tf.exp(x) - 1)


@tf.function
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0, x, alpha * tf.exp(x) - alpha)


@tf.function
def softplus(x):
    return tf.math.log(1 + tf.exp(x))


@tf.function
def leaky_relu(x, alpha=0.3):
    return tf.maximum(x, alpha * x)


@tf.function
def linear(x):
    return x
