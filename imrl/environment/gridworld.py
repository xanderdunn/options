"""Discrete gridworld environment."""

# System
import logging
from collections import namedtuple
from enum import Enum
import random


class Action(Enum):
    '''Possible actions that can be taken in the gridworld.'''
    up = 0
    down = 1
    left = 2
    right = 3

Gridworld = namedtuple("Gridworld", ('size',            # The gridworld is size * size
                                     'failure_rate',    # The probability that any action will fail
                                     'take_action',     # Function to call to apply an action to the environment
                                     'num_actions',     # Total number of possible actions the agent can take
                                     'initial_state'))  # State where the agent begins each episode

Position = namedtuple('Position', ('x', 'y'))
State = namedtuple("State", ('position', 'is_terminal', 'reward'))


def initial_state():
    '''The state the agent starts in at the beginning of each episode.'''
    return State(Position(0, 0), False, None)


def reward(position, environment):
    '''Calculate the reward based on the previous and new positions.'''
    return (is_terminal(position, environment) and 1) or \
           (0)


def is_terminal(position, environment):
    '''Is this position terminal?  That is, is it in the upper left corner?'''
    return position == Position(environment.size - 1, environment.size - 1)


def take_action(current_state, action, environment):
    '''Apply the given action and return the new state.'''
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
        return State(new_pos, is_terminal(new_pos, environment), 0)
    else:
        return State(current_state.position, is_terminal(current_state.position, environment), 0)
