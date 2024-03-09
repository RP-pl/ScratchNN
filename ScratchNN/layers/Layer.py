from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Layer(ABC):

    def __init__(self):
        self.train = False

    @abstractmethod
    @tf.function
    def call(self, input: np.ndarray) -> tf.Tensor:
        """
            Forward pass of the layer
        :param input: data to be processed
        :return: result of the forward pass
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, input_shape: [int]) -> None:
        """
         Initialize all needed variables for the layer.
        :param input_shape: shape of the input tensor
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_shape(self, input_shape: [int]) -> [int]:
        """
            Get the output shape of the layer
        :param input_shape: input shape as list of integers
        :return: output shape of the layer as list of integers
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def get_weights(self) -> [tf.Variable]:
        """
            Return trainable weights of the layer
        :return: list of trainable weights
        """
        raise NotImplementedError

    @abstractmethod
    @tf.function
    def get_regularization_loss(self) -> tf.Tensor:
        """
            Get the fraction of the loss that should be added to the total loss
            :return: Loss coming from regularization
        """
        raise NotImplementedError

    def set_train(self, train: bool) -> None:
        self.train = train
