"""Test the UOM learning."""

# Third party
import numpy as np

# First party
from imrl.agent.option.option import Option
from imrl.agent.policy.policy_fixed import FixedPolicy
from imrl.agent.fa.tabular import TabularFA
from imrl.agent.option.option import Subgoal


def test_uom_update():
    """Is the M matrix updated properly?"""
    option = Option(0, TabularFA(9), FixedPolicy(4, 2), 0.1, 0.99, None, 4)
    assert option.m.shape == (9, 9)
    assert option.u.shape == (9, 9)
    state = 0
    state_prime = 1
    fv = option.fa.evaluate(state)
    fv_prime = option.fa.evaluate(state_prime)
    m_prime = option.update_m(fv, fv_prime, 1.0)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[1, 0] = option.gamma * option.eta
    assert np.array_equal(zeros_matrix, m_prime)
    u_prime = option.update_u(fv)
    zeros_matrix = np.zeros((9, 9))
    zeros_matrix[0, 0] = option.eta
    assert np.array_equal(zeros_matrix, u_prime)
