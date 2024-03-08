from abc import ABC

from ScratchNN.layers import Layer


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def call(self, input):
        raise NotImplementedError

    def build(self, input_shape):
        pass

    def get_output_shape(self, input_shape):
        return input_shape

    def get_weights(self):
        return []

    def get_regularization_loss(self):
        return 0
