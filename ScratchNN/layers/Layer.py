from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

class Layer(ABC):

    def __init__(self):
        self.train = False

    @abstractmethod
    @tf.function
    def call(self, input: np.ndarray) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def build(self, input_shape: [int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_output_shape(self, input_shape: [int]) -> [int]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


    @abstractmethod
    def get_weights(self) -> [tf.Variable]:
        raise NotImplementedError

    @abstractmethod
    @tf.function
    def get_regularization_loss(self) -> tf.Tensor:
        raise NotImplementedError

    def set_train(self, train:bool) -> None:
        self.train = train