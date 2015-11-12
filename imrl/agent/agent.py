"""Managing data structures and algorithms used by an IMRL agent."""

# First party
from imrl.agent.option.option import Option
from imrl.agent.policy.policy_fixed import FixedPolicy

class Agent:

    def __init__(self, policy, fa, num_actions, subgoals):
        self.policy = policy
        self.fa = fa
        self.options = {i: Option(fa, FixedPolicy(num_actions, i)) for i in range(num_actions)}
        self.subgoals = subgoals

    def terminal_update(self, state, action):
        """Called to do any update of the termination state."""
        fv = self.fa.evaluate(state)
        uom = self.options[action].uom
        uom.update_u(fv)

    def update(self, state, action, state_prime):
        """Update an agent with options and return the new agent."""
        tau = 1  # Compute later fomr option stack for non-primitive options
        fv = self.fa.evaluate(state)
        fv_prime = self.fa.evaluate(state_prime)
        uom = self.options[action].uom
        uom.update_m(fv, fv_prime, tau)
        uom.update_u(fv)
        # TODO: check if state_prime is a subgoal and create a new option for it if so

    def create_option(self, subgoal):
        raise NotImplementedError("Should create a new option for the given subgoal.")
