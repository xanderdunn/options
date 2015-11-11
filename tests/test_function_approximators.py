"""Test the environment state function approximators."""

# Third party
import numpy as np

# First party
from imrl.agent.fa.func_approx import tabular_function_approximator
from imrl.environment.gridworld import State, Position
from imrl.agent.fa.rbf import RBF


def test_tabular_function_approximator():
    """Are tabular states represented correctly?"""
    state1 = State(Position(0, 1), False, 0)
    tabular1 = tabular_function_approximator(state1, 9)
    expected1 = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular1, expected1)
    state2 = State(Position(1, 0), False, 0)
    tabular2 = tabular_function_approximator(state2, 9)
    expected2 = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular2, expected2)


def test_rbfs():
    """Test radial basis function approximator."""
    rbfs = RBF(2, 5)
    fv = rbfs.get_features(np.asarray([0, 0]))
    assert abs(1 - fv[0]) < 0.00001
    print(abs(fv[-1]))
    assert abs(fv[-1]) < 0.000001
