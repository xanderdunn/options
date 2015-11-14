# Third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
from imrl.environment.gridworld import Action

# First party
from imrl.agent.policy.policy_vi import VIPolicy

class AgentViz:

    def __init__(self, agent):
        self.agent = agent
        fig, ((self.samples, self.policy), (self.vf, self.reward)) = plt.subplots(2, 2, figsize=(10, 10))

        # Samples plot
        self.samples.set_title('Samples')

        # Policy plot
        self.policy.set_title('Policy')

        # Value function plot
        self.vf.set_title('Value Function')
        div = make_axes_locatable(self.vf)
        self.vf_div = div.append_axes("right", size="5%", pad=0.1)

        # Reward function plot
        self.reward.set_title('Reward Function')
        div = make_axes_locatable(self.reward)
        self.reward_div = div.append_axes("right", size="5%", pad=0.1)

        self.make_grid_samples()

        plt.ion()
        plt.show()

    def make_grid_samples(self):
        segmentation = [np.linspace(0, 1, 10)] * 2
        segmentation = product(*segmentation)
        self.grid_samples = np.asarray([np.asarray(a) for a in segmentation])
        self.state_samples = self.grid_samples

    def update(self):
        self.plot_samples()
        if isinstance(self.agent.policy, VIPolicy):
            self.plot_vf()
            self.plot_reward()
            self.plot_policy()
        self.update_plot_limits()
        plt.draw()
        plt.pause(0.00001)

    def update_plot_limits(self):
        self.policy.set_xlim([-0.1, 1.1])
        self.policy.set_ylim([-0.1, 1.1])

    def plot_samples(self):
        self.samples.cla()
        self.samples.scatter(np.asarray([self.agent.samples[i][0] for i in range(len(self.agent.samples))]),
                             np.asarray([self.agent.samples[i][1] for i in range(len(self.agent.samples))]))

    def plot_vf(self):
        theta = self.agent.policy.vi.theta
        vals = []
        for s in self.state_samples:
            vals.append(np.dot(self.agent.fa.evaluate(s).T, theta))
        self.vf.cla()
        sc = self.vf.scatter(self.grid_samples[:, 0], self.grid_samples[:, 1], s=180, c=vals)
        plt.colorbar(sc, cax=self.vf_div)

    def plot_reward(self):
        reward = self.agent.policy.vi.r
        vals = []
        for s in self.state_samples:
            vals.append(np.dot(self.agent.fa.evaluate(s).T, reward))
        self.reward.cla()
        sc = self.reward.scatter(self.grid_samples[:, 0], self.grid_samples[:, 1], s=180, c=vals)
        plt.colorbar(sc, cax=self.reward_div)

    def plot_policy(self):
        vals = []
        for s in self.state_samples:
            a = int(self.agent.policy.choose_action(s))
            vals.append((a == Action.up and [0, 1]) or
                        (a == Action.down and [0, -1]) or
                        (a == Action.left and [1, 0]) or  # Not sure why positive 1 yields a vector that points left
                        (a == Action.right and [-1, 0]))
        vals = np.asarray(vals)
        self.policy.cla()
        self.policy.quiver(self.grid_samples[:, 0], self.grid_samples[:, 1], vals[:, 0], vals[:, 1],
                           pivot='middle', headwidth=4, headlength=6)
