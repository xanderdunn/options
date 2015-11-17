"""Test the gridworld environment."""

# System
import random

# First party
from imrl.environment.combination_lock import CombinationLock
from imrl.agent.option.option import Subgoal


def test_taking_correct_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    environment = CombinationLock(3, 1, 6, [1, 5, 4], 0.0)

    assert environment.position_from_state(0) == []
    assert environment.position_from_state(1) == [1]
    assert environment.position_from_state(2) == [1, 5]
    assert environment.position_from_state(3) == [1, 5, 4]

    assert not environment.is_terminal()

    # Deterministic tests
    state1 = environment.next_state(environment.initial_state(), 1)
    assert state1 == 1
    assert not environment.is_terminal()

    state2 = environment.next_state(state1, 5)
    assert state2 == 2
    assert not environment.is_terminal()

    state3 = environment.next_state(state2, 4)
    assert state3 == 3
    assert environment.is_terminal()

    # Reset and test taking a wrong action
    environment.reset()
    assert not environment.is_terminal()

    state1 = environment.next_state(environment.initial_state(), 5)
    assert state1 == 0
    assert environment.is_terminal()


def test_subgoals():
    """Test that the right subgoals are set."""
    environment = CombinationLock(3, 1, 6, [1, 5, 4], 0.0)

    assert environment.num_states() == 3

    assert environment.create_subgoals() == [Subgoal(1), Subgoal(2), Subgoal(3)]
