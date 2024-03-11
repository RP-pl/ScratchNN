import numpy as np
import tensorflow as tf

from ScratchNN.layers import Layer
from copy import deepcopy


class Bidirectional(Layer):
    """
        Time distributed layer.
        This layer applies a layer to each time step of the input tensor.
    """

    def __init__(self, layer: Layer):
        super().__init__()
        self.layer_forward = layer
        self.layer_backward = deepcopy(layer)

    @tf.function
    def call(self, input: np.ndarray) -> tf.Tensor:
        return tf.concat([self.layer_forward.call(input), self.layer_backward.call(input[:, ::-1])], axis=-1)


    def build(self, input_shape: [int]) -> None:
        self.layer_backward.build(input_shape)
        self.layer_forward.build(input_shape)

    def get_output_shape(self, input_shape: [int]) -> [int]:
        base_shape = self.layer_forward.get_output_shape(input_shape)
        return base_shape[:-1] + [base_shape[-1] * 2]

    def get_weights(self) -> [tf.Variable]:
        return [*self.layer_forward.get_weights(),*self.layer_backward.get_weights()]

    @tf.function
    def get_regularization_loss(self) -> tf.Tensor:
        return self.layer_forward.get_regularization_loss() +  self.layer_backward.get_regularization_loss()