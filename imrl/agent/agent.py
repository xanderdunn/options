"""Managing data structures and algorithms used by an IMRL agent."""

# Third party
import numpy as np
import random

# First party
from imrl.agent.option.option import Option
from imrl.agent.policy.policy_vi import VIPolicy
from imrl.agent.policy.policy_fixed import FixedPolicy
from imrl.agent.value_iteration import ValueIteration
from imrl.agent.agent_viz import AgentViz
from imrl.agent.agent_viz_disc import AgentVizDisc
from imrl.agent.option.option import Subgoal


class Agent:

    def __init__(self, policy, fa, num_actions, alpha, gamma, eta, zeta, epsilon, plan_iter, sim_samples, sim_steps,
                 retain_theta=True, subgoals=[], samples=[]):
        self.policy = policy
        self.fa = fa
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta
        self.epsilon = epsilon
        self.plan_iterations = plan_iter
        self.sim_samples = sim_samples
        self.sim_steps = sim_steps
        self.subgoals = subgoals
        self.samples = samples
        self.reached_subgoals = []
        self.extrinsic = None
        self.intrinsic = [np.ones((self.fa.num_features, 1))] * num_actions
        self.options = {i: Option(i, fa, FixedPolicy(num_actions, i), eta, gamma, None, num_actions) for i in range(num_actions)}
        self.vi = ValueIteration(-1, self.intrinsic, self, plan_iter, retain_theta=retain_theta, use_options=True, alpha=alpha, gamma=gamma)
        self.vi_policy = VIPolicy(num_actions, self.vi)
        self.option_stack = []
        self.step = 0
        self.viz = None

    def create_visualization(self, discrete=False, gridworld=None):
        num_options = min(4, len(self.subgoals))
        self.viz = AgentVizDisc(self, num_options, gridworld) if discrete else AgentViz(self, num_options)

    def choose_action(self, state):
        """Select an action from the base policy and add to option stack if necessary."""
        fv = self.fa.evaluate(state)
        if not self.option_stack:  # No executing options
            o = self.policy.choose_action(state)
            self.option_stack.append((o, fv, self.step))
        else:
            o, _, _ = self.option_stack[-1]
        while o >= self.num_actions:  # Option selected
            o = self.options[o].policy.choose_action_from_fv(fv)
            self.option_stack.append((o, fv, self.step))
        # print([x[0] for x in self.option_stack])
        return o

    def update(self, state, action, state_prime):
        """Update an agent's models based on a state, action, state tuple."""
        fv = self.fa.evaluate(state)
        fv_prime = self.fa.evaluate(state_prime)
        assert self.option_stack != []
        o_idx, fv_old, start = self.option_stack[-1]
        assert o_idx == action
        tau = self.step - start
        assert tau == 0
        o = self.options[o_idx]
        while o.is_terminal_in_state(state_prime):
            self.option_stack.pop()
            o.update_m(fv_old, fv_prime, tau)
            o.update_u(fv, fv_prime, True)
            if not self.option_stack:
                break
            o_idx, fv_old, start = self.option_stack[-1]
            tau = self.step - start
            o = self.options[o_idx]
        for o, _, _ in self.option_stack:
            self.options[o].update_u(fv, fv_prime, False)
        self.evaluate_sample(state)
        self.evaluate_subgoal(state)
        self.update_intrinsic_reward(state, action)
        self.step += 1

    def update_intrinsic_reward(self, state, action):
        fv = self.fa.evaluate(state)
        self.intrinsic[action] = self.intrinsic[action] + self.zeta * (0 - np.dot(self.intrinsic[action].T, fv)) * fv

    def explore(self):
        self.vi.r = self.intrinsic

    def exploit(self, goal):
        self.extrinsic = [self.fa.evaluate(goal)]
        self.vi.r = self.extrinsic

    def evaluate_sample(self, state):
        """Check if the given state should be added to the state sample set based on distance criterion (epsilon)."""
        if isinstance(state, int):  # Just check set membership for discrete domains.
            if state not in self.samples:
                self.samples.append(state)
                self.samples.sort(reverse=False)
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
                # print('Subgoal reached: ' + str(g.state))
            return

        for g in self.subgoals:
            if np.linalg.norm(np.asarray(g.state - state), 2) <= g.radius and g not in self.reached_subgoals:
                self.create_option(g)
                self.reached_subgoals.append(g)
                # print('Subgoal reached: ' + str(g.state) + ',  ' + str(g.radius))
                break

    def create_option(self, subgoal):
        """Create a new option for the given subgoal with a pseudo reward function and value iteration policy."""
        id = len(self.options)
        vi = ValueIteration(id, [self.fa.evaluate(subgoal.state)], self, self.plan_iterations, alpha=self.alpha, gamma=self.gamma)
        policy = VIPolicy(self.num_actions, vi)
        self.options[id] = Option(id, self.fa, policy, self.eta, self.gamma, subgoal, self.num_actions)
        self.intrinsic.append(np.zeros((self.fa.num_features, 1)))

    def plan(self):
        # Compute option policies
        for i in range(self.num_actions, len(self.options)):
            self.options[i].policy.vi.run()

        # Simulate option policies to learn option models
        for i in range(self.num_actions, len(self.options)):
            for s in random.sample(self.samples, min(self.sim_samples, len(self.samples))):
                self.simulate_policy(self.options[i], s, self.sim_steps)

        # Compute base policy
        self.vi.run()

    def simulate_policy(self, o, start, steps):
        """Simulates an option's policy to provide data for learning M and U.
        Currently assumes primitive option policies."""
        s = self.fa.evaluate(start)
        trajectory = [s]
        for i in range(steps):
            a = o.policy.choose_action_from_fv(s)
            s_prime = self.options[a].get_next_fv(s)
            o.update_u(s, s_prime, o.is_terminal(s_prime))
            self.intrinsic[o.id] = self.intrinsic[o.id] + (np.dot(self.intrinsic[a].T, s) - np.dot(self.intrinsic[o.id].T, s)) * s
            if o.is_terminal(s_prime):
                # print('Sim terminated - ' + str(o.id - self.num_actions + 1))
                for t, state in enumerate(trajectory):
                    o.update_m(state, s_prime, len(trajectory) - t - 1)
                break
            trajectory.append(s_prime)
            s = s_prime