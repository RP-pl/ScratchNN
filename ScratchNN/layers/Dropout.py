from ScratchNN.layers import Layer
import tensorflow as tf

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def call(self, input):
        if self.train:
            dropout_array = tf.random.uniform(input.shape,maxval=1,dtype=tf.float64) > self.rate
            return input * tf.cast(dropout_array, input.dtype)
        else:
            return input

    def build(self, input_shape):
        pass

    def get_output_shape(self, input_shape):
        return input_shape

    def get_weights(self):
        return []

    def get_regularization_loss(self):
        return 0
