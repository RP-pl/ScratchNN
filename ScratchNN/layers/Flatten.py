from ScratchNN.layers import Layer
import numpy as np
import tensorflow as tf


class Flatten(Layer):

    @tf.function
    def call(self, input: np.ndarray) -> tf.Tensor:
        return tf.reshape(input, (input.shape[0], -1))

    def build(self, input_shape: [int]) -> None:
        pass

    def get_output_shape(self, input_shape: [int]) -> [int]:
        return [input_shape[0], np.prod(input_shape[1:])]

    def get_weights(self) -> [tf.Variable]:
        return []

    def get_regularization_loss(self) -> tf.Tensor:
        return tf.constant(0.0, dtype=tf.float64)
