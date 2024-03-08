from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def compile(self, optimizer, loss, input_shape, metrics=[]):
        pass

    @abstractmethod
    def fit(self, X, Y, epochs, batch_size):
        pass

    @abstractmethod
    def predict(self, X):
        pass