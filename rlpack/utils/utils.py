import tensorflow as tf
import numpy as np


def assert_shape(tensor: tf.Tensor, expected_shape):
    """Only work for tensorflow tensor."""
    assert type(expected_shape) is list or type(expected_shape) is tuple
    assert type(tensor) is tf.Tensor
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


def linear_decay(initial_value, final_value, decay_step, current_step):
    if current_step > decay_step:
        return final_value
    return initial_value + (final_value - initial_value) / decay_step * current_step


def exponential_decay(initial_value, rate, decay_step, current_step):
    if current_step > decay_step:
        return initial_value * rate ** decay_step
    return initial_value * rate ** current_step
