import tensorflow as tf
import math
import numpy as np


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])
