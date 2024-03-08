import numpy as np
import tensorflow as tf


def glorot(shape):
    input_units = shape[0]
    output_units = shape[1]
    variance = 1 / (input_units + output_units)
    std_dev = np.sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=(input_units, output_units))
    return weights

def zeros(shape):
    return np.zeros(shape)

def ones(shape):
    return np.ones(shape)

def normal(shape):
    return np.random.normal(size=shape)

def he_normal(shape):
    input_units = shape[0]
    variance = 2 / input_units
    std_dev = np.sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=shape)
    return weights

def he_uniform(shape):
    input_units = shape[0]
    variance = 6 / input_units
    std_dev = np.sqrt(variance)
    weights = np.random.uniform(low=-std_dev, high=std_dev, size=shape)
    return weights


def lecun_normal(shape):
    input_units = shape[0]
    variance = 1 / input_units
    std_dev = np.sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=shape)
    return weights


def lecun_uniform(shape):
    input_units = shape[0]
    variance = 3 / input_units
    std_dev = np.sqrt(variance)
    weights = np.random.uniform(low=-std_dev, high=std_dev, size=shape)
    return weights
