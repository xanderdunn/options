"""Value function-based policy implementation."""

from imrl.agent.policy.policy import Policy


class VIPolicy(Policy):

    def __init__(self, num_actions, vi):
        super(VIPolicy, self).__init__(num_actions)
        self.vi = vi

    def choose_action(self, state):
        return self.vi.get_max_action(state)
