# Third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# First party
from itertools import product
from imrl.environment.gridworld import Action


class AgentViz:

    def __init__(self, agent, num_options):
        self.agent = agent
        self.num_options = num_options
        self.subplots = {}
        fig, subplots = plt.subplots(3, num_options+1, figsize=(17, 12))
        self.subplots['reward'] = subplots[0]
        self.subplots['vf'] = subplots[1]
        self.subplots['policy'] = subplots[2]

        self.vf_divs = []
        self.reward_divs = []
        for i in range(num_options+1):
            self.vf_divs.append(make_axes_locatable(self.subplots['vf'][i]).append_axes("right", size="5%", pad=0.1))
            self.reward_divs.append(make_axes_locatable(self.subplots['reward'][i]).append_axes("right", size="5%", pad=0.1))

        self.make_grid_samples()
        self.set_titles()
        plt.ion()
        plt.show()

    def make_grid_samples(self):
        segmentation = [np.linspace(0, 1, 10)] * 2
        segmentation = product(*segmentation)
        self.grid_samples = np.asarray([np.asarray(a) for a in segmentation])
        self.state_samples = self.grid_samples

    def set_titles(self):
        for i in range(self.num_options + 1):
            self.subplots['reward'][i].set_title('Intrinsic Reward' if i == 0 else 'Option Reward - ' + str(i))
            self.subplots['vf'][i].set_title('Base Value Function' if i == 0 else 'Option VF - ' + str(i))
            self.subplots['policy'][i].set_title('Base Policy' if i == 0 else 'Option Policy - ' + str(i))

    def update(self):
        """Update plots for the base policy, value function, and reward function and for each of three options."""
        # self.plot_samples()
        for i in range(self.num_options + 1):
            if i == 0 or (i > 0 and i + self.agent.num_actions - 1 in self.agent.options):
                self.plot_reward(i)
                self.plot_vf(i)
                self.plot_policy(i)
        self.update_plot_limits()
        self.set_titles()
        plt.draw()
        plt.pause(0.00001)

    def update_plot_limits(self):
        for i in range(self.num_options):
            self.subplots['policy'][i].set_xlim([-0.1, 1.1])
            self.subplots['policy'][i].set_ylim([-0.1, 1.1])

    def plot_samples(self):
        self.samples.cla()
        self.samples.scatter(np.asarray([self.agent.samples[i][0] for i in range(len(self.agent.samples))]),
                             np.asarray([self.agent.samples[i][1] for i in range(len(self.agent.samples))]))

    def plot_reward(self, id):
        reward = self.agent.vi_policy.vi.r if id == 0 else self.agent.options[id + self.agent.num_actions - 1].policy.vi.r
        vals = []
        for s in self.state_samples:
            vals.append(np.dot(self.agent.fa.evaluate(s).T, reward))
        self.subplots['reward'][id].cla()
        sc = self.subplots['reward'][id].scatter(self.grid_samples[:, 0], self.grid_samples[:, 1], s=180, c=vals)
        plt.colorbar(sc, cax=self.reward_divs[id])

    def plot_vf(self, id):
        theta = self.agent.vi_policy.vi.theta if id == 0 else self.agent.options[id + self.agent.num_actions - 1].policy.vi.theta
        vals = []
        for s in self.state_samples:
            vals.append(np.dot(self.agent.fa.evaluate(s).T, theta))
        self.subplots['vf'][id].cla()
        sc = self.subplots['vf'][id].scatter(self.grid_samples[:, 0], self.grid_samples[:, 1], s=180, c=vals)
        plt.colorbar(sc, cax=self.vf_divs[id])

    def plot_policy(self, id):
        policy = self.agent.vi_policy if id == 0 else self.agent.options[id + self.agent.num_actions - 1].policy
        vals = []
        for s in self.state_samples:
            a = int(policy.choose_action(s))
            vals.append((a == Action.up and [0, 1]) or
                        (a == Action.down and [0, -1]) or
                        (a == Action.left and [1, 0]) or  # Not sure why positive 1 yields a vector that points left
                        (a == Action.right and [-1, 0]))
        vals = np.asarray(vals)
        self.subplots['policy'][id].cla()
        self.subplots['policy'][id].quiver(self.grid_samples[:, 0], self.grid_samples[:, 1], vals[:, 0], vals[:, 1],
                           pivot='middle', headwidth=4, headlength=6)
