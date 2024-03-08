import numpy as np


def glorot(shape):
    input_units = shape[0]
    output_units = shape[1]
    variance = 1 / (input_units + output_units)
    std_dev = np.sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=(input_units, output_units))
    return weights
