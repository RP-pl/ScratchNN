import numpy as np
from keras.backend import conv1d
from typing_extensions import deprecated

from ScratchNN.activations import linear
from ScratchNN.initializations import glorot
from ScratchNN.layers import Layer
from ScratchNN.util import valid
import tensorflow as tf


@deprecated("Use Conv1D instead. This class is terribly slow")
class _Conv1D(Layer):
    def __init__(self,filters,kernel_size,strides=1,padding=valid,activation=linear,initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super(_Conv1D,self).__init__()
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
        self.kernels = tf.Variable(tf.random.normal([self.filters, self.kernel_size, input_shape[-1]], dtype=tf.float64),dtype=tf.float64)
        self.bias = tf.Variable(tf.zeros(self.filters,dtype=tf.float64),dtype=tf.float64)


    @tf.function
    def call(self,inputs):
        inputs = tf.pad(inputs,[[0,0],[self.kernel_size-1,0],[0,0]])
        output = self._convolve_with_channels(inputs,self.kernels)+self.bias
        return output

    @tf.function
    def _reorganize_input(self, input: np.ndarray) -> tf.Tensor:
        """
            Reorganize input by timestamp rather than by batch
        """
        return tf.transpose(input, [2, 0, 1])

    @tf.function
    def _convolve_with_channels(self,inputs: np.ndarray, filters: np.ndarray):
        """
            Return 1D convoulution of inputs with filter_data
            Input format: (batch, sequence, channels)
            Filter format: (filters,kernel, channels)
            Output format: (batch, sequence, filters)
        """
        outputs = []
        kernel_size = filters.shape[1]
        for j in range(inputs.shape[1] - kernel_size + 1):
            window = inputs[:, j:j + kernel_size, :]
            @tf.function
            def conv(x):
                return tf.reduce_sum(x * window, axis=2)
            convolution = tf.reduce_sum(tf.map_fn(conv, filters), axis=2)
            outputs.append(convolution)
        return tf.transpose(tf.stack(outputs, axis=2), [1, 2, 0])

    def get_weights(self) -> [tf.Variable]:
        return [self.kernels,self.bias]

    def get_output_shape(self, input_shape: [int]) -> [int]:
        return [input_shape[0],input_shape[1],self.filters]
    def get_regularization_loss(self) -> tf.Tensor:
        if self.kernel_regularizer is not None and self.bias_regularizer is not None:
            return self.kernel_regularizer(self.kernels) + self.bias_regularizer(self.bias)
        elif self.kernel_regularizer is not None:
            return self.kernel_regularizer(self.kernels)
        elif self.bias_regularizer is not None:
            return self.bias_regularizer(self.bias)
        else:
            return 0