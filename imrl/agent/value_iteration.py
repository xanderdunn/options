"""Value iteration algorithm."""

# System
import random

# Third party
import numpy as np

class ValueIteration:

    def __init__(self, id, reward_function, agent, iterations, use_options=False, alpha=0.1, gamma=0.99):
        self.id = id
        self.agent = agent
        self.r = reward_function
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.theta = np.zeros((agent.fa.num_features, 1))
        self.use_options = use_options

    def update_r(self):
        self.r = self.r

    def run(self):
        """Run value iteration for the given number of iterations starting from a zero-initialized value function"""
        theta = np.zeros((self.agent.fa.num_features, 1))
        for i in range(self.iterations):
            theta = self.sweep(theta)
        self.theta = theta
        return theta

    def sweep(self, theta):
        """Adjust the current value function estimate theta by performing a full backup."""
        for s in self.agent.samples:
            theta = self.backup(theta, s)
        return theta

    def backup(self, theta, s):
        """Get the maximum Bellman residual over all options."""
        fv = self.agent.fa.evaluate(s)
        if self.use_options:
            max_value = max([self.get_residual(theta, o, fv) for o in self.agent.options.values() if o.id != self.id])
        else:
            max_value = max([self.get_residual(theta, o, fv) for o in self.agent.options.values() if o.id < self.agent.num_actions])
        delta = (self.alpha * (max_value - np.dot(fv.T, theta))) * fv
        return theta + delta

    def get_residual(self, theta, o, fv):
        """Calculate the scalar product that is used in both the theta and policy calculations."""
        return o.get_return(self.r, fv) + self.gamma * np.dot(o.get_next_fv(fv).T, theta)

    def get_max_action(self, fv):
        if self.use_options:
            values = [self.get_residual(self.theta, o, fv) for o in self.agent.options.values() if o.id != self.id]
            # print(values)
        else:
            values = [self.get_residual(self.theta, o, fv) for o in self.agent.options.values() if o.id < self.agent.num_actions]
        max_value = max(values)
        max_value_actions = [i for i, x in enumerate(values) if x == max_value]

        # If max value actions include both options and primitives, only select from the primitives.
        primitives = [i for i in max_value_actions if i < self.agent.num_actions]
        if len(primitives) > 0:
            max_value_actions = primitives
        # else:
        #     print('Option chosen: ' + str(values))
        return random.choice(max_value_actions)
