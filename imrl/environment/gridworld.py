"""Discrete gridworld environment."""

# System
from collections import namedtuple
from enum import Enum
import random

# First party
from imrl.utils.linear_algebra import sparse_one_hot_vector


class Action(Enum):
    """Possible actions that can be taken in the gridworld."""
    up = 0
    down = 1
    left = 2
    right = 3

Gridworld = namedtuple("Gridworld", ('size',                # The gridworld is size * size
                                     'failure_rate',        # The probability that any action will fail
                                     'take_action',         # Function to call to apply an action to the environment
                                     'num_actions',         # Total number of possible actions the agent can take
                                     'initial_state',       # State where the agent begins each episode
                                     'reward_vector',       # The reward function represented as a vector
                                     'exhaustive_states'))  # Exhaustive list of possible states for a discrete tabular environment

Position = namedtuple('Position', ('x', 'y'))
State = namedtuple("State", ('position', 'is_terminal', 'reward'))


def gridworld_discrete(size, failure_rate):
    """Return a discrete gridworld environment."""
    return Gridworld(size, failure_rate, take_action, 4, initial_state, reward_vector(size * size), exhaustive_states(size * size))


def exhaustive_states(num_states):
    return [sparse_one_hot_vector(num_states, i) for i in range(num_states)]


def reward_vector(num_states):
    return sparse_one_hot_vector(num_states, num_states - 1)


def initial_state():
    """The state in which the agent starts at the beginning of each episode."""
    return State(Position(0, 0), False, 0)


def reward(position, is_terminal):
    """Calculate the reward based on the previous and new positions."""
    return (is_terminal and 1) or (0)


def is_terminal(position, environment):
    """Is this position terminal?  That is, is it in the upper left corner?"""
    coordinate = environment.size - 1
    return position == Position(coordinate, coordinate)


def take_action(current_state, size, action, environment):
    """Apply the given action and return the new state."""
    posx = current_state.position.x
    posy = current_state.position.y
    if random.random() > environment.failure_rate:
        mapped_action = Action(action)
        tentative_pos = (mapped_action == Action.up and Position(posx, posy + 1)) or \
                        (mapped_action == Action.down and Position(posx, posy - 1)) or \
                        (mapped_action == Action.left and Position(posx + 1, posy)) or \
                        (mapped_action == Action.right and Position(posx - 1, posy))
        new_pos = ((tentative_pos.x >= 0 and tentative_pos.y >= 0 and
                    tentative_pos.y < environment.size and tentative_pos.x < environment.size) and tentative_pos) or \
                  (current_state.position)
        terminal = is_terminal(new_pos, environment)
        return State(new_pos, terminal, reward(new_pos, terminal))
    else:
        return current_state
