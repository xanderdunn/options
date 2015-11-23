"""Random policy implementation."""

# Third party
import numpy as np

# First party
from imrl.agent.policy.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, num_actions):
        super(RandomPolicy, self).__init__(num_actions)

    def choose_action(self, state):
        return self.choose_action_from_fv(state)

    def choose_action_from_fv(self, state):
        return np.random.randint(self.num_actions)

class RandomOptionPolicy(RandomPolicy):

    def __init__(self, agent, use_options):
        super(RandomPolicy, self).__init__(agent.num_actions)
        self.agent = agent
        self.use_options = use_options

    def choose_action(self, state):
        return self.choose_action_from_fv(state)

    def choose_action_from_fv(self, stfvate):
        return np.random.randint(len(self.agent.options) if self.use_options else self.num_actions)
