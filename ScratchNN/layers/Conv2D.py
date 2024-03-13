from ScratchNN.activations import linear
from ScratchNN.initializations import glorot
from ScratchNN.layers import Layer

import tensorflow as tf

from ScratchNN.layers.ConvBase import ConvBase


class Conv2D(ConvBase):
    def __init__(self, filters, kernel_size, strides=(1,1), activation=linear, initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super(Conv2D,self).__init__(filters, kernel_size, strides, activation, initializer, kernel_regularizer, bias_regularizer)