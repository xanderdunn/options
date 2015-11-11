"""Continuous gridworld environment."""

# System
from collections import namedtuple
from enum import Enum
import numpy as np

class Action(Enum):
    """Possible actions that can be taken in the gridworld."""
    up = 0
    down = 1
    left = 2
    right = 3

ContinuousGridworld = namedtuple("ContinuousGridworld", (
                                    'move_mean',           # Mean of distance moved when taking an action
                                    'move_sd',                    # Standard deviation of distance moved when taking an action
                                    'take_action',         # Function to call to apply an action to the environment
                                    'num_actions',         # Total number of possible actions the agent can take
                                    'initial_state',       # State where the agent begins each episode
                                    'reward_center',       # The center point of the goal region
                                    'reward_radius'))      # The radius of the goal region

Position = namedtuple('Position', ('x', 'y'))
State = namedtuple("State", ('position', 'is_terminal', 'reward'))


def gridworld_continuous(move_mean, move_sd):
    """Return a discrete gridworld environment."""
    return ContinuousGridworld(move_mean, move_sd, take_action, 4, initial_state, Position(0.95, 0.95), 0.02)


def initial_state():
    """The state in which the agent starts at the beginning of each episode."""
    return State(Position(max(np.random.normal(0, 0.01), 0), max(np.random.normal(0, 0.01), 0)), False, 0)


def reward(position, environment):
    """Calculate the reward based on the previous and new positions."""
    return (is_terminal(position, environment) and 1) or 0


def is_terminal(position, environment):
    """Is this position terminal?  That is, is it in the upper left corner?"""
    return np.linalg.norm(np.asarray([position.x - environment.reward_center.x, position.y - environment.reward_center.y]), 2) < environment.reward_radius


def take_action(current_state, action, environment):
    """Apply the given action and return the new state."""
    posx = current_state.position.x
    posy = current_state.position.y
    mapped_action = Action(action)
    noise = np.random.normal(0, environment.move_sd)
    move = np.random.normal(environment.move_mean, environment.move_sd)
    tentative_pos = (mapped_action == Action.up and Position(posx + noise, posy + move)) or \
                    (mapped_action == Action.down and Position(posx + noise, posy - move)) or \
                    (mapped_action == Action.left and Position(posx + move, posy + noise)) or \
                    (mapped_action == Action.right and Position(posx - move, posy + noise))
    new_pos = Position(min(1, max(tentative_pos.x, 0)), min(1, max(tentative_pos.y, 0)))
    return State(new_pos, is_terminal(new_pos, environment), reward(new_pos, environment))

if __name__ == '__main__':
    env = ContinuousGridworld(0.05, 0.01, take_action, 4, initial_state, Position(0.95, 0.95), 0.02)
    s = env.initial_state()
    steps = 0
    while not is_terminal(s.position, env):
        s = take_action(s, np.random.randint(4), env)
        print(s.position)
        steps +=1
    print(steps)
