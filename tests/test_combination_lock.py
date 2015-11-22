"""Test the gridworld environment."""

# System
import random

# First party
from imrl.environment.combination_lock import CombinationLock
from imrl.agent.option.option import Subgoal
from imrl.environment.gridworld import GridPosition


def test_taking_correct_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    environment = CombinationLock(3, 3, 6, 0.0, [1, 5, 4])

    assert environment.actions_from_state(0) == []
    assert environment.actions_from_state(1) == [1]
    assert environment.actions_from_state(2) == [1, 5]
    assert environment.actions_from_state(3) == [1, 5, 4]

    assert not environment.is_terminal([])

    # Deterministic tests
    state1 = environment.next_state(environment.initial_state(), 1)
    assert state1 == 1
    assert not environment.is_terminal(environment.actions_from_state(state1))

    state2 = environment.next_state(state1, 5)
    assert state2 == 2
    assert not environment.is_terminal(environment.actions_from_state(state2))

    state3 = environment.next_state(state2, 4)
    assert state3 == 3

    assert environment.is_terminal(environment.actions_from_state(state3) + [2])
    state4 = environment.next_state(state3, 4)
    assert state4 == 0
    assert not environment.is_terminal(environment.actions_from_state(state4))

    assert environment.is_terminal([2])


def test_subgoals():
    """Test that the right subgoals are set."""
    environment = CombinationLock(3, 3, 6, 0.0, [1, 5, 4])

    assert environment.num_states() == 9

    assert environment.create_subgoals() == [Subgoal(3), Subgoal(6), Subgoal(9)]


def test_grid_position():
    """Are the grid positions correct?"""
    environment = CombinationLock(3, 3, 6, [1, 5, 4], 0.0)
    assert environment.grid_position_from_state(2) == GridPosition(2, 0)
    assert environment.grid_position_from_state(3) == GridPosition(0, 1)

