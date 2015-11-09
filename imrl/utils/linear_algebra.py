"""Convenience linear algebra functions."""

import numpy as np


def sparse_one_hot_vector(length, position):
    """Return a \"one-hot\" float vector of given length that is all 0's except for a 1 in the given position."""
    zero_vector = np.zeros((length, 1))
    zero_vector[position, 0] = 1.0
    return zero_vector
