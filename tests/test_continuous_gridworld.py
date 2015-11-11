"""Test the continuous gridworld environment."""

# System
import random
from functools import partial

# IMRL
from imrl.environment.gridworld_continuous import take_action, GridworldContinuous, Action, Position, is_terminal, initial_state
from imrl.environment.gridworld import reward
from tests.tools import ratio_test

# Third Party
import numpy as np


def test_taking_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    env = GridworldContinuous(0.05, 0.01, take_action, 4, initial_state, Position(0.95, 0.95), 0.02, is_terminal, reward)
    start = env.initial_state()
    ratio = ratio_test(lambda state: np.linalg.norm(np.asarray([state.position.x - start.position.x,
                                                                state.position.y - (start.position.y + env.move_mean)]),
                                                       2) < env.move_sd*2,
                       partial(take_action, start, Action.up, env), 10000)
    assert 0.85 < ratio
    steps = 0
    s = env.initial_state()
    while not is_terminal(s.position, env):
        s = take_action(s, np.random.randint(4), env)
        steps += 1
    assert steps < 20000


def test_termination():
    """Does the environment terminate in the correct state?"""
    environment = GridworldContinuous(0.05, 0.01, take_action, 4, initial_state, Position(0.95, 0.95), 0.02, is_terminal, reward)
    assert not is_terminal(Position(0, 0), environment)
    assert is_terminal(Position(0.96, 0.94), environment)
