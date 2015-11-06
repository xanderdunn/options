"""Test the environment state function approximators."""

import numpy as np
from imrl.agent.fa.func_approx import tabular_function_approximator
from scipy.sparse import csc_matrix


def test_tabular_function_approximator():
    """Are tabular states represented correctly?"""
    tabular = tabular_function_approximator(3.0, 3)
    expected = csc_matrix(np.array([[0.0], [0.0], [1.0]]))
    assert (tabular != expected).nnz == 0
