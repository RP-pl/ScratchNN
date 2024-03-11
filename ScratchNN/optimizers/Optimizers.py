from abc import ABC, abstractmethod

import tensorflow as tf


class Optimizer(ABC):

    def __init__(self, lr=0.01):
        self.lr = lr
        self.tape = tf.GradientTape(persistent=True)

    @abstractmethod
    @tf.function
    def apply_gradients(self, grads, weights):
        pass

class SGD(Optimizer):

    def __init__(self, lr=1e-5):
        """
        Stochastic Gradient Descent Optimizer
        :param lr: Learning Rate
        """
        super().__init__(lr=lr)

    @tf.function
    def apply_gradients(self, grads, weights):
        for grad, weight in zip(grads, weights):
            weight.assign_sub(self.lr * grad)


class AdaGrad(Optimizer):

    def __init__(self, lr=1e-5):
        """
        Adaptive Gradient Optimizer
        :param lr: Learning Rate
        """
        super().__init__(lr=lr)
        self.s = None

    @tf.function
    def apply_gradients(self, grads, weights):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
        new_s = []
        for grad, weights, prev_s in zip(grads, weights, self.s):
            s = prev_s + tf.math.multiply(grad, grad)
            weights.assign_sub(tf.math.divide(self.lr, tf.math.sqrt(s + 1e-5)) * grad)
            new_s.append(s)
        self.s = new_s


class RMSProp(Optimizer):

    def __init__(self, lr=1e-5, beta=0.9):
        """
        Root Mean Square Propagation Optimizer
        :param lr: learning rate
        :param beta: forgetting factor
        """
        super().__init__(lr=lr)
        self.beta = beta
        self.s = None

    @tf.function
    def apply_gradients(self, grads, weights):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
        new_s = []
        for grad, weights, prev_s in zip(grads, weights, self.s):
            s = self.beta * prev_s + (1 - self.beta) * tf.math.multiply(grad, grad)
            weights.assign_sub(tf.math.divide(self.lr, tf.math.sqrt(s + 1e-5)) * grad)
            new_s.append(s)
        self.s = new_s


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        """
        Adam Optimizer
        :param beta1: forgetting factor for first moment
        :param beta2: forgetting factor for second moment
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.s = None
        self.v = None
        self.t = 0

    @tf.function
    def apply_gradients(self, grads: [tf.Tensor], weights: [tf.Tensor]):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
            self.v = [tf.zeros_like(w) for w in weights]
        new_s = []
        new_v = []
        self.t += 1
        for grad, weights, prev_s, prev_v in zip(grads, weights, self.s, self.v):
            v = self.beta1 * prev_v + (1 - self.beta1) * grad
            s = self.beta2 * prev_s + (1 - self.beta2) * tf.math.multiply(grad, grad)
            v_hat = v / (1 - self.beta1 ** self.t)
            s_hat = s / (1 - self.beta2 ** self.t)
            weights.assign_sub(self.lr * v_hat / tf.sqrt(s_hat + 1e-5))
            new_v.append(v)
            new_s.append(s)
        self.s = new_s
        self.v = new_v


class AdaMax(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        """
        AdaMax Optimizer
        :param beta1: forgetting factor for first moment
        :param beta2: forgetting factor for second moment
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.s = None
        self.v = None
        self.t = 0

    @tf.function
    def apply_gradients(self, grads: [tf.Tensor], weights: [tf.Tensor]):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
            self.v = [tf.zeros_like(w) for w in weights]
        new_s = []
        new_v = []
        self.t += 1
        for grad, weights, prev_s, prev_v in zip(grads, weights, self.s, self.v):
            v = self.beta1 * prev_v + (1 - self.beta1) * grad
            s = tf.math.maximum(self.beta2 * prev_s, tf.abs(grad))
            v_hat = v / (1 - self.beta1 ** self.t)
            s_hat = s / (1 - self.beta2 ** self.t)
            weights.assign_sub(self.lr * v_hat / tf.sqrt(s_hat + 1e-5))
            new_v.append(v)
            new_s.append(s)
        self.s = new_s
        self.v = new_v


class Nadam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        """
        Nadam Optimizer
        :param beta1: forgetting factor for first moment
        :param beta2: forgetting factor for second moment
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.s = None
        self.v = None
        self.t = 0
    @tf.function
    def apply_gradients(self, grads, weights):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
            self.v = [tf.zeros_like(w) for w in weights]
        new_s = []
        new_v = []
        self.t += 1
        for grad, weights, prev_s, prev_v in zip(grads, weights, self.s, self.v):
            v = self.beta1 * prev_v + (1 - self.beta1) * grad
            s = self.beta2 * prev_s + (1 - self.beta2) * tf.math.multiply(grad, grad)
            v_hat = v / (1 - self.beta1 ** self.t)
            s_hat = s / (1 - self.beta2 ** self.t)
            v_bar = self.beta1 *  v_hat + (1-self.beta1) * grad
            weights.assign_sub(self.lr * v_bar / tf.sqrt(s_hat + 1e-5))
            new_v.append(v)
            new_s.append(s)
        self.s = new_s
        self.v = new_v