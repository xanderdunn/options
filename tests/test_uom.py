"""Test the UOM learning."""

# Third party
import numpy as np

# First party
from imrl.agent.uom import update_m, update_u, uom_primitive
from imrl.agent.fa.func_approx import tabular_function_approximator


def test_m_update():
    """Is the M matrix updated properly?"""
    uom = uom_primitive(9)
    assert uom.initial_m.shape == (9, 9)
    fv = tabular_function_approximator(0.0, 9)
    fv_prime = tabular_function_approximator(1.0, 9)
    m_prime = update_m(uom, uom.initial_m, fv, fv_prime)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[1, 0] = uom.gamma
    assert np.array_equal(zeros_matrix, m_prime.toarray())


def test_u_update():
    """Is the U matrix updated properly?"""
    uom = uom_primitive(9)
    assert uom.initial_u.shape == (9, 9)
    fv = tabular_function_approximator(0.0, 9)
    u_prime = update_u(uom, uom.initial_u, fv)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[0, 0] = 1
    assert np.array_equal(zeros_matrix, u_prime.toarray())
