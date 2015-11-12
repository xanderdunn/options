"""Fixed policy implementation."""

# Third party
import numpy as np

# First party
from imrl.agent.policy.policy import Policy


class FixedPolicy(Policy):

    def __init__(self, num_actions, action):
        super(FixedPolicy, self).__init__(num_actions)
        self.action = action

    def choose_action(self, state):
        return self.action