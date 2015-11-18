"""Test the environment state function approximators."""

# Third party
import numpy as np

# First party
from imrl.agent.fa.tabular import TabularFA
from imrl.environment.gridworld import GridPosition
from imrl.agent.fa.rbf import RBF


def test_tabular_function_approximator():
    """Are tabular states represented correctly?"""
    state1 = 3
    tabular1 = TabularFA(9, 4)
    expected1 = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular1.evaluate(state1), expected1)
    state2 = 1
    tabular2 = TabularFA(9, 4)
    expected2 = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    assert np.array_equal(tabular2.evaluate(state2), expected2)


def test_rbfs():
    """Test radial basis function approximator."""
    rbfs = RBF(2, 5, 4)
    fv = rbfs.evaluate(np.asarray([0, 0]))
    assert abs(1 - fv[0]) < 0.00001
    print(abs(fv[-1]))
    assert abs(fv[-1]) < 0.000001
