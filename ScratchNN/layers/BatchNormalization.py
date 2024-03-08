import numpy as np

from ScratchNN.layers import Layer
import tensorflow as tf


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3, axis=-1, gamma_regularizer=None, beta_regularizer=None):
        super().__init__()
        self.gamma_regularizer = gamma_regularizer
        self.beta_regularizer = beta_regularizer
        self.momentum = momentum
        self.epsilon = epsilon
        self.axis = axis
        self.gamma = None
        self.beta = None
        self.mean = None
        self.variance = None
        self.running_mean = None
        self.running_variance = None
        self.trainable = True
        self.train = True

    def build(self, input_shape):
        self.gamma = tf.Variable(np.ones(input_shape[self.axis]), dtype=tf.float64)
        self.beta = tf.Variable(np.zeros(input_shape[self.axis]), dtype=tf.float64)
        self.running_mean = tf.Variable(np.zeros(input_shape[self.axis]), dtype=tf.float64, trainable=False)
        self.running_variance = tf.Variable(np.ones(input_shape[self.axis]), dtype=tf.float64, trainable=False)

    @tf.function
    def call(self, batch):
        if self.train:
            mean = tf.reduce_mean(batch, axis=0)
            variance = tf.reduce_mean((batch - mean) ** 2, axis=0)
            self.running_mean.assign(self.running_mean * self.momentum + mean * (1 - self.momentum))
            self.running_variance.assign(self.running_variance * self.momentum + variance * (1 - self.momentum))
        else:
            mean = self.running_mean
            variance = self.running_variance

        h = (batch - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * h + self.beta

    def get_output_shape(self, input_shape):
        return input_shape

    def get_weights(self):
        return [self.gamma, self.beta]

    @tf.function
    def get_regularization_loss(self):
        if self.gamma_regularizer is not None and self.beta_regularizer is not None:
            return self.gamma_regularizer(self.gamma) + self.beta_regularizer(self.beta)
        elif self.gamma_regularizer is not None:
            return self.gamma_regularizer(self.gamma)
        elif self.beta_regularizer is not None:
            return self.beta_regularizer(self.beta)
        return 0

    def set_train(self, train):
        self.train = train
