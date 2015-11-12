"""Managing data structures and algorithms used by an IMRL agent."""

# Third party
import numpy as np

# First party
from imrl.agent.option.option import Option
from imrl.agent.policy.policy_fixed import FixedPolicy

class Agent:

    def __init__(self, policy, fa, num_actions, alpha, gamma, eta, epsilon, samples=[], subgoals=[]):
        self.policy = policy
        self.fa = fa
        self.options = {i: Option(fa, FixedPolicy(num_actions, i), eta, gamma) for i in range(num_actions)}
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.samples = samples
        self.subgoals = subgoals

    def terminal_update(self, state, action):
        """Called to do any update of the termination state."""
        fv = self.fa.evaluate(state)
        uom = self.options[action].uom
        uom.update_u(fv)
        self.evaluate_sample(state)
        # self.evaluate_subgoal(state)

    def update(self, state, action, state_prime):
        """Update an agent with options and return the new agent."""
        tau = 1  # Compute later from option stack for non-primitive options
        fv = self.fa.evaluate(state)
        fv_prime = self.fa.evaluate(state_prime)
        uom = self.options[action].uom
        uom.update_m(fv, fv_prime, tau)
        uom.update_u(fv)
        self.evaluate_sample(state)
        # self.evaluate_subgoal(state)

    def evaluate_sample(self, state):
        """Check whether the given state should be added to the state sample set based on distance citerion (epsilon)."""
        if isinstance(state, int):  # Just check set membership for discrete domains.
            if state not in self.samples:
                self.samples.append(state)
            return

        # TODO replace sample list with KD-tree
        add = True
        for s in self.samples:
            if np.linalg.norm(np.asarray(s - state), 2) <= self.epsilon:
                add = False
                break
        if add:
            self.samples.append(state)

    def evaluate_subgoal(self, state):
        """Check whether the given state is a subgoal for an as yet uncreated option and create one if so."""
        raise NotImplementedError("Should create a new option for the given state if it is a subgoal.")

