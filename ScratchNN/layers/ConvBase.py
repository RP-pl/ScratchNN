from ScratchNN.activations import linear
from ScratchNN.initializations import glorot
from ScratchNN.layers import Layer
import tensorflow as tf


class ConvBase(Layer):
    """
        Base for all convolutional layers.
    """
    def __init__(self, filters, kernel_size, strides=None, activation=linear, initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super(ConvBase,self).__init__()
        self.strides = strides
        self.bias = None
        self.kernels = None
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
    def build(self,input_shape):
        values = tf.cast(self.initializer([*self.kernel_size,input_shape[-1],self.filters]),dtype=tf.float64)
        self.kernels = tf.Variable(values,dtype=tf.float64)
        self.bias = tf.Variable(self.initializer([self.filters]),dtype=tf.float64)
    @tf.function
    def call(self,inputs):
        output = tf.nn.convolution(inputs, self.kernels, strides=self.strides, padding='VALID')
        return self.activation(output) + self.bias

    def get_weights(self) -> [tf.Variable]:
        return [self.kernels,self.bias]

    def get_output_shape(self, input_shape: [int]) -> [int]:
        output = [input_shape[0]]
        for i in range(len(input_shape)-2):
            output.append((input_shape[i+1]-self.kernel_size[i])//self.strides[i]+1)
        output.append(self.filters)
        return output
    def get_regularization_loss(self) -> tf.Tensor:
        if self.kernel_regularizer is not None and self.bias_regularizer is not None:
            return self.kernel_regularizer(self.kernels) + self.bias_regularizer(self.bias)
        elif self.kernel_regularizer is not None:
            return self.kernel_regularizer(self.kernels)
        elif self.bias_regularizer is not None:
            return self.bias_regularizer(self.bias)
        else:
            return 0