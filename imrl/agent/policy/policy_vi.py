"""Value function-based policy implementation."""

from imrl.agent.policy.policy import Policy


class VIPolicy(Policy):

    def __init__(self, num_actions, vi):
        super(VIPolicy, self).__init__(num_actions)
        self.vi = vi

    def choose_action(self, state):
        fv = self.vi.agent.fa.evaluate(state)
        return self.choose_action_from_fv(fv)

    def choose_action_from_fv(self, fv):
        return self.vi.get_max_action(fv)
