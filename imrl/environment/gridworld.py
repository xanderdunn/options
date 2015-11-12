"""Discrete gridworld environment."""

# System
from collections import namedtuple
from enum import Enum
import random

# First party
from imrl.utils.linear_algebra import one_hot_vector
from imrl.environment.environment import Environment


class Action(Enum):
    """Possible actions that can be taken in the gridworld."""
    up = 0
    down = 1
    left = 2
    right = 3

GridworldState = namedtuple('GridworldState', ('x', 'y'))


class Gridworld(Environment):
    """Discrete gridworld implementation. Assumes square shape."""

    def __init__(self, size, failure_rate, num_actions=4):
        super(Gridworld, self).__init__(num_actions)
        self.size = size
        self.failure_rate = failure_rate

    def num_states(self):
        return self.size * self.size

    def exhaustive_states(self):
        return [one_hot_vector(self.num_states(), i) for i in range(self.num_states())]

    def reward_vector(self):
        return one_hot_vector(self.num_states(), self.num_states() - 1)

    def reward(self, state):
        """Return 1 if transitioning into corner (size-1, size-1)"""
        return 1.0 if state == GridworldState(self.size-1, self.size-1) else 0.0

    def initial_state(self):
        """Always start the agent in the corner (0,0)."""
        return GridworldState(0, 0)

    def is_terminal(self, state):
        """The corner (size-1, size-1) is a terminal state."""
        return state == GridworldState(self.size-1, self.size-1)

    def next_state(self, state, action):
        """Apply the given action and return the next state."""
        next_state = state
        if random.random() > self.failure_rate:
            mapped_action = Action(action)
            tentative_state = (mapped_action == Action.up and GridworldState(state.x, state.y + 1)) or \
                            (mapped_action == Action.down and GridworldState(state.x, state.y - 1)) or \
                            (mapped_action == Action.left and GridworldState(state.x + 1, state.y)) or \
                            (mapped_action == Action.right and GridworldState(state.x - 1, state.y))
            if 0 <= tentative_state.x < self.size and 0 <= tentative_state.y < self.size:
                next_state = tentative_state
        return next_state
