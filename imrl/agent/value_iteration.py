"""Value iteration algorithm."""

# System
import random

# Third party
import numpy as np


class ValueIteration:

    def __init__(self, id, reward_functions, agent, iterations, retain_theta=True, use_options=False, alpha=0.1, gamma=0.99):
        self.id = id
        self.agent = agent
        self.r = reward_functions
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.theta = np.zeros((agent.fa.num_features, 1))
        self.use_options = use_options
        self.retain_theta = retain_theta

    def run(self):
        """Run value iteration for the given number of iterations starting from a zero-initialized value function"""
        theta = self.theta if self.retain_theta else np.zeros((self.agent.fa.num_features, 1))
        for i in range(self.iterations):
            theta = self.sweep(theta)
        self.theta = theta
        return theta

    def sweep(self, theta):
        """Adjust the current value function estimate theta by performing a full backup."""
        if self.use_options:
            option_set = [o for i, o in self.agent.options.items() if i != self.id]
        else:
            option_set = [o for i, o in self.agent.options.items() if i < self.agent.num_actions]
        if self.id >= self.agent.num_actions:
            samples = self.agent.options[self.id].get_init_set()
        else:
            samples = self.agent.samples
        for s in samples:
            theta = self.backup(theta, s, option_set)
        return theta

    def backup(self, theta, s, options):
        """Get the maximum Bellman residual over all options."""
        fv = self.agent.fa.evaluate(s)
        max_value = max([self.get_value(theta, o, fv) for o in options])
        delta = (self.alpha * (max_value - np.dot(fv.T, theta))) * fv
        return theta + delta

    def get_value(self, theta, o, fv):
        """Calculate the scalar product that is used in both the theta and policy calculations."""
        r = self.r[0] if len(self.r) == 1 else self.r[o.id]
        return o.get_return(r, fv) + self.gamma * np.dot(o.get_next_fv(fv).T, theta)

    def get_max_action(self, fv):
        if self.use_options:
            values = [self.get_value(self.theta, o, fv) for o in self.agent.options.values() if o.id != self.id]
        else:
            values = [self.get_value(self.theta, o, fv) for o in self.agent.options.values() if o.id < self.agent.num_actions]
        max_value = max(values)
        max_value_actions = [i for i, x in enumerate(values) if x == max_value]

        # If max value actions include both options and primitives, only select from the primitives.
        primitives = [i for i in max_value_actions if i < self.agent.num_actions]
        if len(primitives) > 0:
            max_value_actions = primitives
        # else:
        #     print('Option chosen: ' + str(values))
        return random.choice(max_value_actions)
