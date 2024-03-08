from abc import ABC, abstractmethod

import numpy as np

import ScratchNN.optimizers


class Model(ABC):

    @abstractmethod
    def compile(self, optimizer:ScratchNN.optimizers.Optimizer, loss, input_shape:[int], metrics=[]):
        pass

    @abstractmethod
    def fit(self, X, Y, epochs, batch_size, shuffle=True, validation_data:[np.ndarray]=None):
        pass

    @abstractmethod
    def predict(self, X:np.ndarray):
        pass