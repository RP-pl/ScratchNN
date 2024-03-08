import tensorflow as tf


@tf.function
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


@tf.function
def tanh(x):
    return tf.tanh(x)


@tf.function
def softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
