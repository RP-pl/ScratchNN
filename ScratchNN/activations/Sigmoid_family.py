import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def tanh(x):
    return tf.tanh(x)

def softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)