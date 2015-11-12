"""Test the continuous gridworld environment."""

# System
import random
from functools import partial

# IMRL
from imrl.environment.gridworld_continuous import GridworldContinuous, Action, GridPosition
from tests.tools import ratio_test

# Third Party
import numpy as np


def test_taking_actions():
    """Does the environment correctly change the state when told to take an action with and without stochasticity?"""
    random.seed()
    env = GridworldContinuous(0.05, 0.01)
    start = env.initial_state()
    ratio = ratio_test(lambda state: np.linalg.norm(np.asarray([state[0] - start[0], state[1] - (start[1] + env.move_mean)]), 2) < env.move_sd * 2,
                       partial(env.next_state, start, Action.up), 10000)
    assert 0.7 < ratio
    steps = 0
    s = env.initial_state()
    while not env.is_terminal(s):
        s = env.next_state(s, np.random.randint(4))
        steps += 1
    assert steps < 20000


def test_termination():
    """Does the environment terminate in the correct state?"""
    env = GridworldContinuous(0.05, 0.01)
    assert not env.is_terminal(GridPosition(0, 0))
    assert env.is_terminal(GridPosition(0.97, 0.98))
