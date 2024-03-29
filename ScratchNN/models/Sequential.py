import pickle
from builtins import callable
import tensorflow as tf

from ScratchNN.layers.Layer import Layer
from ScratchNN.models.Model import Model


class Sequential(Model):
    """
    Sequential model that stacks layers on top of each other.
    """
    def __init__(self, layers=None):
        self.metrics = []
        self.layers: [Layer] = layers if layers is not None else []
        self.loss = None
        self.optimizer = None
        self.weights = []

    def add(self, l):
        if isinstance(l, list):
            self.layers.extend(l)
        elif isinstance(l, Layer):
            self.layers.append(l)
        else:
            raise ValueError("Layer must be of type Layer or list of Layer")

    def compile(self, optimizer, loss, input_shape, metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        for layer in self.layers:
            layer.build(input_shape)
            self.weights.extend(layer.get_weights())
            input_shape = layer.get_output_shape(input_shape)
        if type(metrics) == list:
            self.metrics.extend(metrics)
        elif callable(metrics):
            self.metrics.append(metrics)
        else:
            raise ValueError("Metrics must be of type list of metrics or function")

    def fit(self, X, Y, epochs, batch_size, shuffle=True, validation_data=None):
        for layer in self.layers:
            layer.set_train(True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            if shuffle:
                indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
                indices = tf.random.shuffle(indices)
                X = tf.gather(X, indices)
                Y = tf.gather(Y, indices)

            for i in range(0, len(X), batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = Y[i:i + batch_size]
                self._fit_batch(x_batch, y_batch)
            print(f"Epoch loss: {self.loss(Y, self.predict(X))}")
            if validation_data is not None:
                X_val, Y_val = validation_data
                for metric in self.metrics:
                    print(f"Epoch {metric.__name__}: {metric(Y_val, self.predict(X_val))}")
            else:
                for metric in self.metrics:
                    print(f"Epoch {metric.__name__}: {metric(Y, self.predict(X))}")

        for layer in self.layers:
            layer.set_train(False)

    @tf.function
    def _fit_batch(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_batch)
            loss = self.loss(y_batch, y_pred)
            for layer in self.layers:
                loss += tf.cast(layer.get_regularization_loss(), dtype=tf.float64)
        self.optimizer.apply_gradients(tape.gradient(loss, self.weights), self.weights)

    @tf.function
    def predict(self, X):
        Y_pred = self.layers[0](X)
        for layer in self.layers[1:]:
            Y_pred = layer(Y_pred)
        return Y_pred

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> Model:
        with open(path, 'rb') as f:
            return pickle.load(f)