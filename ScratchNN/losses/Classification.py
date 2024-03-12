import numpy as np
import tensorflow as tf


@tf.function
def cross_entropy(y_true, y_pred):
    """
    Cross Entropy Loss
    :param y_true: true labels in one hot encoding
    :param y_pred: predicted labels in one hot encoding
    :return: Cross Entropy loss between y_true and y_pred
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.clip_by_value(y_true, 1e-10, 1)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))


@tf.function
def kl_divergence(y_true, y_pred):
    """
    Kullback-Leibler Divergence Loss
    :param y_true: true labels in one hot encoding
    :param y_pred: true labels in one hot encoding
    :return: KL Divergence loss between y_true and y_pred
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.clip_by_value(y_true, 1e-10, 1)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred))


@tf.function
def hinge(y_true, y_pred):
    """
    Hinge Loss
    :param y_true: true labels in one hot encoding
    :param y_pred: true labels in one hot encoding
    :return: hinge loss between y_true and y_pred
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.clip_by_value(y_true, 1e-10, 1)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1)
    return tf.maximum(0, 1 - y_true * y_pred)

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross Entropy Loss
    :param y_true: true labels in one hot encoding
    :param y_pred: true labels in one hot encoding
    :return: binary cross entropy loss between y_true and y_pred
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.clip_by_value(y_true, 1e-10, 1)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))