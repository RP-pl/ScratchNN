from abc import ABC, abstractmethod

import tensorflow as tf

class Optimizer(ABC):

    def __init__(self, lr=0.01):
        self.lr = lr
    @abstractmethod
    def apply_gradients(self, grads, weights):
        pass


class SGD(Optimizer):

    def __init__(self, lr=1e-5):
        super().__init__(lr=lr)

    def apply_gradients(self, grads, weights):
        for grad, weight in zip(grads, weights):
            weight.assign_sub(self.lr*grad)

class AdaGrad(Optimizer):

    def __init__(self,lr=1e-5):
        super().__init__(lr=lr)
        self.s = None

    def apply_gradients(self, grads, weights):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
        new_s = []
        for grad, weights,prev_s in zip(grads, weights,self.s):
            s = prev_s+tf.math.multiply(grad, grad)
            weights.assign_sub(tf.math.divide(self.lr, tf.math.sqrt(s + 1e-5)) * grad)
            new_s.append(s)
        self.s = new_s

class RMSProp(Optimizer):

    def __init__(self, lr=1e-5, beta=0.9):
        super().__init__(lr=lr)
        self.beta = beta
        self.s = None

    def apply_gradients(self, grads, weights):
        if self.s is None:
            self.s = [tf.zeros_like(w) for w in weights]
        new_s = []
        for grad, weights,prev_s in zip(grads, weights,self.s):
            s = self.beta*prev_s + (1-self.beta)*tf.math.multiply(grad, grad)
            weights.assign_sub(tf.math.divide(self.lr, tf.math.sqrt(s + 1e-5)) * grad)
            new_s.append(s)
        self.s = new_s