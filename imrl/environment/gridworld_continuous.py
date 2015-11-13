"""Continuous gridworld environment."""

# Third party
import numpy as np

# First party
from imrl.environment.gridworld import GridPosition, Action
from imrl.environment.environment import Environment


class GridworldContinuous(Environment):

    def __init__(self, move_mean=0.05, move_sd=0.01, reward_center=np.asarray([1.0, 1.0]), reward_radius=0.1, num_actions=4):
        super(GridworldContinuous, self).__init__(num_actions)
        self.move_mean = move_mean
        self.move_sd = move_sd
        self.reward_center = reward_center
        self.reward_radius = reward_radius

    def reward_vector(self):
        return self.reward_center

    def reward(self, state):
        return 1.0 if self.is_terminal(state) else 0.0

    def initial_state(self):
        """The state in which the agent starts at the beginning of each episode."""
        x_pos = max(np.random.normal(0, 0.01), 0)
        y_pos = max(np.random.normal(0, 0.01), 0)
        return np.asarray([x_pos, y_pos])

    def is_terminal(self, state):
        """Is this position terminal?"""
        diff = [state[0] - self.reward_center[0], state[1] - self.reward_center[1]]
        return np.linalg.norm(np.asarray(diff), 2) < self.reward_radius

    def next_state(self, state, action):
        """Apply the given action and return the new state."""
        mapped_action = Action(action)
        noise = np.random.normal(0, self.move_sd)
        move = np.random.normal(self.move_mean, self.move_sd)
        tentative_pos = (mapped_action == Action.up and GridPosition(state[0] + noise, state[0] + move)) or \
                        (mapped_action == Action.down and GridPosition(state[0] + noise, state[0] - move)) or \
                        (mapped_action == Action.left and GridPosition(state[0] + move, state[0] + noise)) or \
                        (mapped_action == Action.right and GridPosition(state[0] - move, state[0] + noise))
        return np.asarray([min(1, max(tentative_pos.x, 0)), min(1, max(tentative_pos.y, 0))])
