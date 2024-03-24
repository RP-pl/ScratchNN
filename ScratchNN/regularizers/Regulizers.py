from abc import ABC, abstractmethod
import tensorflow as tf


class Regulizer(ABC):
    """
    Abstract class for regularizers.
    Regularizers are used to add penalties to the loss function to prevent overfitting.
    """

    @abstractmethod
    def call(self, weights):
        pass

    def __call__(self, weights):
        return self.call(weights)


class L1(Regulizer):
    def __init__(self, alpha=0.01):
        """
        Regularizer that enforces L1 regularization.
        :param alpha: penalty multiplier
        """
        self.alpha = alpha

    @tf.function
    def call(self, weights):
        return self.alpha * tf.reduce_sum(tf.abs(weights))


class L2(Regulizer):
    def __init__(self, alpha=0.01):
        """
        Regularizer that enforces L2 regularization.
        :param alpha: penalty multiplier
        """
        self.alpha = alpha

    @tf.function
    def call(self, weights):
        return self.alpha * tf.reduce_sum(tf.square(weights))


class L1L2(Regulizer):
    def __init__(self, l1=0.01, l2=0.01, alpha=1, beta=1):
        """
        Regularizer that enforces L1 and L2 regularization.
        \alpha * L1(weights) + \beta * L2(weights)
        :param l1: amount of L1 regularization
        :param l2: amount of L2 regularization
        :param alpha: multiplier for L1
        :param beta: multiplier for L2
        """
        self.l1 = L1(l1)
        self.l2 = L2(l2)
        self.alpha = alpha
        self.beta = beta

    @tf.function
    def call(self, weights):
        return self.alpha * self.l1(weights) + self.beta * self.l2(weights)


class Orthogonal(Regulizer):
    """
    Regulizer that enforces orthogonality of the weights.
    Input needs to be 2D tensor.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    @tf.function
    def call(self, weights):
        if len(weights.shape) != 2:
            raise ValueError("Weights must be 2D tensor. Got shape: ", weights.shape)
        return self.alpha * tf.reduce_sum(
            tf.matmul(weights, weights, transpose_b=True) - tf.eye(weights.shape[0], dtype=tf.float64))
