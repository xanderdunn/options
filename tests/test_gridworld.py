"""Test the gridworld environment."""

# System
import random
from functools import partial

# IMRL
from imrl.environment.gridworld import GridPosition, Gridworld, Action
from tests.tools import ratio_test


def test_taking_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    env = Gridworld(4, 0.0)

    # Deterministic tests
    assert env.next_state(env.initial_state(), Action.up) == env.state_from_grid_position(GridPosition(0, 1))
    assert env.next_state(env.initial_state(), Action.down) == env.state_from_grid_position(GridPosition(0, 0))
    assert env.next_state(env.initial_state(), Action.left) == env.state_from_grid_position(GridPosition(1, 0))
    assert env.next_state(env.initial_state(), Action.right) == env.state_from_grid_position(GridPosition(0, 0))

    # Stochastic tests
    env.failure_rate = 0.1
    assert ratio_test(lambda state: state == env.state_from_grid_position(GridPosition(0, 0)), partial(env.next_state, env.initial_state(), Action.right), 10000) == 1.0
    ratio = ratio_test(lambda state: state == env.state_from_grid_position(GridPosition(0, 0)), partial(env.next_state, env.initial_state(), Action.up), 10000)
    assert ratio > 0.09 and ratio < 0.11


def test_termination():
    """Does the environment terminate in the correct state?"""
    env = Gridworld(4, 0)
    assert not env.is_terminal(env.state_from_grid_position(GridPosition(0, 0)))
    assert env.is_terminal(env.state_from_grid_position(GridPosition(3, 3)))
