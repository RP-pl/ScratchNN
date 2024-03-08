import tensorflow as tf


@tf.function
def mse(y_true, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    y_true = tf.cast(y_true, dtype=tf.float64)
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.function
def mape(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100


@tf.function
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
