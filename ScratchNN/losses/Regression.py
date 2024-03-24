import tensorflow as tf


@tf.function
def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error.
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean squared error between predicted and true values
    """
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    y_true = tf.cast(y_true, dtype=tf.float64)
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error.
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean absolute error between predicted and true values
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.function
def mape(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE).
    :param y_true: true values
    :param y_pred: predicted values
    :return: mape score between predicted and true values
    """
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100


@tf.function
def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error.
    :param y_true: true values
    :param y_pred: predicted values
    :return: root mean squared error between predicted and true values
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
