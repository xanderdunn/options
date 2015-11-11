"""Test the environment state function approximators."""

# Third party
import numpy as np

# First party
from imrl.agent.fa.func_approx import tabular_function_approximator
from imrl.agent.fa.rbf import RBF


def test_tabular_function_approximator():
    """Are tabular states represented correctly?"""
    tabular = tabular_function_approximator(3, 9)
    expected = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular, expected)
    tabular_two = tabular_function_approximator(1, 9)
    expected_two = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular_two, expected_two)

def test_rbfs():
    """Test radial basis function approximator."""
    rbfs = RBF(2, 5)
    fv = rbfs.get_features(np.asarray([0, 0]))
    assert abs(1 - fv[0]) < 0.00001
    print(abs(fv[-1]))
    assert abs(fv[-1]) < 0.000001

