import numpy as np

def _parse_tensor(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()
    return y_true, y_pred
def accuracy(y_true, y_pred):
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def precision(y_true, y_pred):
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    false_positives = np.sum(np.round(np.clip(y_pred - y_true, 0, 1)), axis=0)
    return true_positives / (true_positives + false_positives + 1e-7)

def recall(y_true, y_pred):
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    false_negatives = np.sum(np.round(np.clip(y_true - y_pred, 0, 1)), axis=0)
    return true_positives / (true_positives + false_negatives + 1e-7)

def f1(y_true, y_pred):
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-7)

def mape(y_true, y_pred):
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true, y_pred):
    """
    Compute the R^2 score.
    R^2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.
    The perfect score is 1.0.
    :param y_true: true values
    :param y_pred: predicted values
    :return: R^2 score between predicted and true values
    """
    y_true, y_pred = _parse_tensor(y_true, y_pred)
    return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
