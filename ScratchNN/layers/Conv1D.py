import numpy as np

from ScratchNN.activations import linear
from ScratchNN.initializations import glorot
from ScratchNN.layers import Layer

import tensorflow as tf

from ScratchNN.layers.ConvBase import ConvBase


class Conv1D(ConvBase):
    """
        Convolutional 1D layer.
    """
    def __init__(self,filters,kernel_size,strides=1,activation=linear,initializer=glorot, kernel_regularizer=None, bias_regularizer=None):
        super(Conv1D,self).__init__(filters, [kernel_size], [strides], activation, initializer, kernel_regularizer, bias_regularizer)