import numpy as np
import tensorflow as tf

from ScratchNN.layers import Layer


class TimeDistributed(Layer):
    """
        Time distributed layer.
        This layer applies a layer to each time step of the input tensor.
    """

    def __init__(self, layer: Layer):
        super().__init__()
        self.layer = layer

    @tf.function
    def call(self, input: np.ndarray) -> tf.Tensor:
        return tf.map_fn(lambda x: self.layer.call(x), input)


    def build(self, input_shape: [int]) -> None:
        self.layer.build(input_shape[1:])

    def get_output_shape(self, input_shape: [int]) -> [int]:
        return input_shape


    def get_weights(self) -> [tf.Variable]:
        return self.layer.get_weights()

    @tf.function
    def get_regularization_loss(self) -> tf.Tensor:
        return self.layer.get_regularization_loss()