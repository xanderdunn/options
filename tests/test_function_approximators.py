"""Test the environment state function approximators."""

# Third party
import numpy as np

# First party
from imrl.agent.fa.func_approx import tabular_function_approximator


def test_tabular_function_approximator():
    """Are tabular states represented correctly?"""
    tabular = tabular_function_approximator(3, 9)
    expected = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular, expected)
    tabular_two = tabular_function_approximator(1, 9)
    expected_two = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular_two, expected_two)
