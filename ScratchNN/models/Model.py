from abc import ABC, abstractmethod

import numpy as np

import ScratchNN.optimizers


class Model(ABC):
    """
    Base class for all models.
    Model class is an abstract class that represents a model to be trained and evaluated.
    """

    @abstractmethod
    def compile(self, optimizer: ScratchNN.optimizers.Optimizer, loss, input_shape: [int], metrics=[]):
        """
        Compile the model with the given optimizer, loss function, input shape and metrics.
        :param optimizer: Optimizer to use for training the model
        :param loss: Loss function to use for training the model
        :param input_shape: shape of the input data
        :param metrics: list of metrics to evaluate the model
        :return:
        """
        pass

    @abstractmethod
    def fit(self, X, Y, epochs, batch_size, shuffle=True, validation_data: [np.ndarray] = None) -> None:
        """
        Train the model on the given dataset.
        :param X: input data
        :param Y: true values
        :param epochs: number of epochs to train the model
        :param batch_size: size of the training batch
        :param shuffle: whether to shuffle the dataset before each epoch
        :param validation_data: dataset to validate the model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        Predict the output for the given input data.
        :param X: input data
        :return: predicted values
        """
        pass
