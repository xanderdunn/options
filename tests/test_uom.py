"""Test the UOM learning."""

# Third party
import numpy as np

# First party
from imrl.agent.uom import update_m, update_u, uom_primitive
from imrl.environment.gridworld import State, Position
from imrl.agent.fa.func_approx import tabular_function_approximator


def test_m_update():
    """Is the M matrix updated properly?"""
    uom = uom_primitive(9, 1.0, 0.999)
    assert uom.m.shape == (9, 9)
    state = State(Position(0, 0), False, 0)
    state_prime = State(Position(1, 0), False, 0)
    fv = tabular_function_approximator(state, 9)
    fv_prime = tabular_function_approximator(state_prime, 9)
    m_prime = update_m(uom, fv, fv_prime, 1.0)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[1, 0] = uom.descriptor.gamma
    assert np.array_equal(zeros_matrix, m_prime)


def test_u_update():
    """Is the U matrix updated properly?"""
    uom = uom_primitive(9, 1.0, 0.999)
    assert uom.u.shape == (9, 9)
    state = State(Position(0, 0), False, 0)
    fv = tabular_function_approximator(state, 9)
    u_prime = update_u(uom, fv)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[0, 0] = 1
    assert np.array_equal(zeros_matrix, u_prime)
