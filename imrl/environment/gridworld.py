"""Discrete gridworld environment."""

# System
from collections import namedtuple
from enum import IntEnum
import random
# First party
from imrl.utils.linear_algebra import one_hot_vector
from imrl.environment.environment import Environment
from imrl.agent.option.option import Subgoal


class Action(IntEnum):
    """Possible actions that can be taken in the gridworld."""
    up = 0
    down = 1
    left = 2
    right = 3


GridPosition = namedtuple('GridPosition', ('x', 'y'))


class Gridworld(Environment):
    """Discrete gridworld implementation."""

    def __init__(self, width, height, failure_rate, num_actions=4):
        super(Gridworld, self).__init__(num_actions)
        self.width = width
        self.height = height
        self.failure_rate = failure_rate

    def num_states(self):
        return self.width * self.height

    def exhaustive_states(self):
        return [one_hot_vector(self.num_states(), i) for i in range(self.num_states())]

    def grid_position_from_state(self, state):
        """Returns a coordinate pair based on the given state vector."""
        return GridPosition(state % self.width, state // self.width)

    def state_from_grid_position(self, position):
        """Returns a state index from a grid position."""
        return int(position.x + self.width * position.y)

    def create_subgoals(self):
        return [Subgoal(self.state_from_grid_position(GridPosition(self.width//2, self.height//2))),
                Subgoal(self.state_from_grid_position(GridPosition(self.width-1, 0))),
                Subgoal(self.state_from_grid_position(GridPosition(0, self.height-1)))]

    def reward_vector(self):
        return self.num_states() - 1

    def reward(self, state):
        """Return 1 if transitioning into corner (size-1, size-1)"""
        return 1.0 if self.is_terminal(state) else 0.0

    def initial_state(self):
        """Always start the agent in the corner (0,0)."""
        return 0

    def is_terminal(self, state):
        """The corner (size-1, size-1) is a terminal state."""
        return state == self.num_states() - 1

    def next_state(self, state, action):
        """Apply the given action and return the next state."""
        position = self.grid_position_from_state(state)
        next_state = position
        if random.random() > self.failure_rate:
            mapped_action = Action(action)
            tentative_state = (mapped_action == Action.up and GridPosition(position.x, position.y + 1)) or \
                              (mapped_action == Action.down and GridPosition(position.x, position.y - 1)) or \
                              (mapped_action == Action.left and GridPosition(position.x - 1, position.y)) or \
                              (mapped_action == Action.right and GridPosition(position.x + 1, position.y))
            if 0 <= tentative_state.x < self.width and 0 <= tentative_state.y < self.height:
                next_state = tentative_state
        return self.state_from_grid_position(next_state)
