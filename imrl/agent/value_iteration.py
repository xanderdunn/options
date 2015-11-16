"""Value iteration algorithm."""

# System
import random

# Third party
import numpy as np

class ValueIteration:

    def __init__(self, id, reward_function, agent, iterations, alpha=0.1, gamma=0.99):
        self.id = id
        self.agent = agent
        self.r = reward_function
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.theta = np.zeros((agent.fa.num_features, 1))

    def update_r(self):
        self.r = self.r

    def run(self):
        """Run value iteration for the given number of iterations starting from a zero-initialized value function"""
        theta = self.theta  # np.zeros((self.agent.fa.num_features, 1))
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
        max_value = max([self.get_residual(theta, o, fv) for o in self.agent.options.values()])
        delta = (self.alpha * (max_value - np.dot(fv.T, theta))) * fv
        return theta + delta

    def get_residual(self, theta, o, fv):
        """Calculate the scalar product that is used in both the theta and policy calculations."""
        return np.dot(self.r.T, np.dot(o.uom.u, fv)) + self.gamma * np.dot(np.dot(o.uom.m, fv).T, theta)

    def get_max_action(self, s):
        fv = self.agent.fa.evaluate(s)
        values = [self.get_residual(self.theta, o, fv) for o in self.agent.options.values() if o.id != self.id]
        max_value = max(values)
        max_value_actions = [i for i, x in enumerate(values) if x == max_value]

        # If max value actions include both options and primitives, only select from the primitives.
        primitives = [i for i in max_value_actions if i < self.agent.num_actions]
        if len(primitives) > 0:
            max_value_actions = primitives
        # else:
        #     print('Option chosen: ' + str(values))
        return random.choice(max_value_actions)
