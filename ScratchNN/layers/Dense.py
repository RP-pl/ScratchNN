import numpy as np
import tensorflow as tf

import ScratchNN.activations
from ScratchNN.layers.Layer import Layer
from ScratchNN.initializations.Initializations import glorot
class Dense(Layer):

    def __init__(self, neurons, activation=ScratchNN.activations.linear,initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super().__init__()
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.b:tf.Variable = None
        self.w:tf.Variable = None
        self.weights:[tf.Variable] = None
        self.neurons = neurons
        self.activation = activation
        self.initializer = initializer

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Input shape must be a 2D tensor")
        self.w = tf.Variable(self.initializer((input_shape[-1],self.neurons)), dtype=tf.float64, name="dense_weights")
        self.b = tf.Variable(np.random.rand(self.neurons), dtype=tf.float64, name="dense_biases")
        self.weights = [self.w, self.b]
    def call(self, input):
        output = tf.matmul(input, self.w) + self.b
        return self.activation(output)

    def get_regularization_loss(self):
        if self.kernel_regularizer is not None and self.bias_regularizer is not None:
            return self.kernel_regularizer(self.w) + self.bias_regularizer(self.b)
        elif self.kernel_regularizer is not None:
            return self.kernel_regularizer(self.w)
        elif self.bias_regularizer is not None:
            return self.bias_regularizer(self.b)
        return 0
    def get_output_shape(self, input_shape):
        return *input_shape[:-1], self.neurons

    def get_weights(self):
        return self.weights

    def set_train(self, train):
        pass
