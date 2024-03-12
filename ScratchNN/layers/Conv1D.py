import numpy as np

from ScratchNN.activations import linear
from ScratchNN.initializations import glorot
from ScratchNN.layers import Layer
from ScratchNN.util import valid

import tensorflow as tf

class Conv1D(Layer):
    def __init__(self,filters,kernel_size,strides=1,padding=valid,activation=linear,initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super(Conv1D,self).__init__()
        self.bias = None
        self.kernels = None
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
    def build(self,input_shape):
        self.kernels = tf.Variable(tf.random.normal([self.kernel_size,input_shape[-1],self.filters], dtype=tf.float64))

    def call(self,inputs):
        output = self.conv(inputs)
        return output


    def conv(self,inputs):
        output = tf.nn.convolution(inputs,self.kernels, strides=self.strides, padding='VALID',data_format="NWC",dilations=[1])
        return output

    def get_weights(self) -> [tf.Variable]:
        return [self.kernels]

    def get_output_shape(self, input_shape: [int]) -> [int]:
        return [input_shape[0],(input_shape[1]-self.kernel_size)//self.strides+1,self.filters]
    def get_regularization_loss(self) -> tf.Tensor:
        if self.kernel_regularizer is not None and self.bias_regularizer is not None:
            return self.kernel_regularizer(self.kernels) + self.bias_regularizer(self.bias)
        elif self.kernel_regularizer is not None:
            return self.kernel_regularizer(self.kernels)
        elif self.bias_regularizer is not None:
            return self.bias_regularizer(self.bias)
        else:
            return 0