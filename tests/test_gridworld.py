"""Test the gridworld environment."""

# System
import random
from functools import partial

# IMRL
from imrl.environment.gridworld import take_action, State, Gridworld, Action, Position, is_terminal, initial_state, reward_vector, exhaustive_states
from tests.tools import ratio_test


def test_taking_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    deterministic_environment = Gridworld(4, 0.0, take_action, 4, initial_state, reward_vector(4), exhaustive_states(4))
    assert take_action(initial_state(), deterministic_environment.size, Action.up, deterministic_environment) == State(Position(0, 1), False, 0)
    assert take_action(initial_state(), deterministic_environment.size, Action.down, deterministic_environment) == State(Position(0, 0), False, 0)
    assert take_action(initial_state(), deterministic_environment.size, Action.left, deterministic_environment) == State(Position(1, 0), False, 0)
    assert take_action(initial_state(), deterministic_environment.size, Action.right, deterministic_environment) == State(Position(0, 0), False, 0)
    stochastic_environment = Gridworld(4, 0.1, take_action, 4, initial_state, reward_vector(4), exhaustive_states(4))
    assert ratio_test(lambda state: state == State(Position(0, 0), False, 0), partial(take_action, initial_state(), deterministic_environment.size, Action.right, stochastic_environment), 10000) == 1.0
    ratio = ratio_test(lambda state: state == State(Position(0, 0), False, 0), partial(take_action, initial_state(), deterministic_environment.size, Action.up, stochastic_environment), 10000)
    assert ratio > 0.09 and ratio < 0.11


def test_termination():
    """Does the environment terminate in the correct state?"""
    environment = Gridworld(4, 0.1, take_action, 4, initial_state, reward_vector(4), exhaustive_states(4))
    assert not is_terminal(Position(0, 0), environment)
    assert is_terminal(Position(3, 3), environment)
