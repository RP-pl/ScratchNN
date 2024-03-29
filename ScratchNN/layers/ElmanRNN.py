import numpy as np

from ScratchNN.layers import Layer
from ScratchNN.initializations import glorot
from ScratchNN.activations import tanh, linear
import tensorflow as tf


class ElmanRNN(Layer):
    """
        Fully connected Elman RNN layer.
        Uses the following equations:
        .. math::
            h_{n} = \phi(X @ W + h_{n-1} @ W_h + b) \n
            y = \psi(h)
    """

    def __init__(self, neurons, activation=linear, recurrent_activation=tanh, return_sequences=False,
                 kernel_initializer=glorot, recurrent_initializer=glorot, kernel_regularizer=None,
                 recurrent_regularizer=None):
        super(ElmanRNN, self).__init__()
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_reguarlizer = recurrent_regularizer
        self.state_weight = None
        self.weights = None
        self.neurons = neurons
        self.b = None
        self.w = None
        self.return_sequences = return_sequences
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.activation = activation
        self.recurrent_activation = recurrent_activation

    @tf.function
    def call(self, input: np.ndarray):
        """
        :param input: input tensor
        :return: output tensor
        """
        X = self._reorganize_input(input)
        X = tf.cast(X, dtype=tf.float64)
        state = self._do_recurrent_step(X[0], np.zeros((X.shape[1], self.neurons)).astype(np.float64))
        output = self._do_recurrence(X, state)
        return output

    @tf.function
    def _do_recurrent_step(self, input: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        """
            Perform a single recurrent step
        """
        Y = self.recurrent_activation(tf.matmul(input, self.w) + tf.matmul(state, self.state_weight) + self.b)
        return Y

    @tf.function
    def _do_recurrence(self, input: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        sequences = []
        for i in range(1, input.shape[0]): # iterate over time steps
            Y = self._do_recurrent_step(input[i], state)
            if self.return_sequences:
                sequences.append(Y)
            state = Y
        if self.return_sequences:
            return self.activation(self._reorganize_input(tf.convert_to_tensor(sequences)))
        else:
            return self.activation(state)

    @tf.function
    def _reorganize_input(self, input: np.ndarray) -> tf.Tensor:
        """
            Reorganize input by timestamp rather than by batch
        """
        return tf.transpose(input, [1, 0, 2])

    def build(self, input_shape: [int]) -> None:
        if len(input_shape) != 3:
            raise ValueError("Input shape must be a 3D tensor")
        n_inputs = input_shape[-1]
        n_neurons = self.neurons
        self.w = tf.Variable(self.kernel_initializer((n_inputs, n_neurons)), dtype=tf.float64, name="dense_weights")
        self.state_weight = tf.Variable(self.recurrent_initializer((n_neurons, n_neurons)), dtype=tf.float64,
                                        name="recurrent_weights")
        self.b = tf.Variable(np.random.rand(n_neurons), dtype=tf.float64, name="dense_biases")
        self.weights = [self.w, self.b, self.state_weight]

    def get_output_shape(self, input_shape: [int]) -> [int]:
        if not self.return_sequences:
            return input_shape[:1] + [self.neurons]
        else:
            return input_shape[:2][::-1] + [self.neurons]

    def get_weights(self) -> [tf.Variable]:
        return self.weights

    @tf.function
    def get_regularization_loss(self) -> tf.Tensor:
        # Written this way so that tf.function can work properly
        if self.kernel_regularizer is not None and self.recurrent_reguarlizer is not None:
            return self.kernel_regularizer(self.w) + self.recurrent_reguarlizer(self.b)
        elif self.kernel_regularizer is not None:
            return self.kernel_regularizer(self.w)
        elif self.recurrent_reguarlizer is not None:
            return self.recurrent_reguarlizer(self.b)
        return 0
