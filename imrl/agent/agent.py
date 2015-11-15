"""Managing data structures and algorithms used by an IMRL agent."""

# Third party
import numpy as np

# First party
from imrl.agent.option.option import Option
from imrl.agent.policy.policy_vi import VIPolicy
from imrl.agent.policy.policy_fixed import FixedPolicy
from imrl.agent.value_iteration import ValueIteration
from imrl.agent.agent_viz import AgentViz
from imrl.agent.agent_viz_disc import AgentVizDisc
from imrl.agent.option.option import Subgoal


class Agent:

    def __init__(self, policy, fa, irf, num_actions, alpha, gamma, eta, epsilon, plan_iter, samples=[], subgoals=[]):
        self.policy = policy
        self.fa = fa
        self.intrinsic_reward = irf
        self.plan_iterations = plan_iter
        self.num_actions = num_actions
        self.options = {i: Option(i, fa, FixedPolicy(num_actions, i), eta, gamma, None) for i in range(num_actions)}
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.samples = samples
        self.subgoals = subgoals
        self.reached_subgoals = []
        self.viz = None
        self.vi = ValueIteration(-1, irf, self, plan_iter, alpha, gamma)
        self.vi_policy = VIPolicy(num_actions, self.vi)

    def create_visualization(self, discrete=False, gridworld=None):
        num_options = min(4, len(self.subgoals))
        self.viz = AgentVizDisc(self, num_options, gridworld) if discrete else AgentViz(self, num_options)

    def terminal_update(self, state, action, state_prime):
        """Called to do any update of the termination state."""
        self.update(state, action, state_prime)

    def update(self, state, action, state_prime):
        """Update an agent with options and return the new agent."""
        tau = 1  # Compute later from option stack for non-primitive options
        fv = self.fa.evaluate(state)
        fv_prime = self.fa.evaluate(state_prime)
        uom = self.options[action].uom
        uom.update_m(fv, fv_prime, tau)
        uom.update_u(fv)
        self.evaluate_sample(state)
        self.evaluate_subgoal(state)

    def evaluate_sample(self, state):
        """Check whether the given state should be added to the state sample set based on distance criterion (epsilon)."""
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
        if isinstance(state, int):  # Just check set membership for discrete domains.
            g = Subgoal(state)
            if g in self.subgoals and g not in self.reached_subgoals:
                self.create_option(g)
                self.reached_subgoals.append(g)
                print('Subgoal reached: ' + str(g.state))
            return

        for g in self.subgoals:
            if np.linalg.norm(np.asarray(g.state - state), 2) <= g.radius and g not in self.reached_subgoals:
                self.create_option(g)
                self.reached_subgoals.append(g)
                print('Subgoal reached: ' + str(g.state) + ',  ' + str(g.radius))
                break

    def create_option(self, subgoal):
        """Create a new option for the given subgoal with a pseudo reward function and value iteration policy."""
        id = len(self.options)
        vi = ValueIteration(id, subgoal.state, self, self.plan_iterations, self.alpha, self.gamma)
        policy = VIPolicy(self.num_actions, vi)
        self.options[id] = Option(id, self.fa, policy, self.alpha, self.gamma, subgoal)

    def plan(self):
        for i in range(self.num_actions, len(self.options)):
            self.options[i].policy.vi.run()
        self.vi.run()