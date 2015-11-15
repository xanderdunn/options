# Third party
import numpy as np
import matplotlib.pyplot as plt

# First party
from imrl.agent.agent_viz import AgentViz
# from imrl.environment.gridworld import Action

class AgentVizDisc(AgentViz):

    def __init__(self, agent, num_options, gridworld):
        self.gridworld = gridworld
        super(AgentVizDisc, self).__init__(agent, num_options)

    def make_grid_samples(self):
        self.grid_samples = np.zeros((self.gridworld.size ** 2, 2))
        self.state_samples = []
        for i in range(self.gridworld.size ** 2):
            p = self.gridworld.grid_position_from_state(i)
            self.grid_samples[i][0] = p.x
            self.grid_samples[i][1] = p.y
            self.state_samples.append(i)

    def update_plot_limits(self):
        for i in range(self.num_options):
            self.subplots['policy'][i].set_xlim([-1, self.gridworld.size])
            self.subplots['policy'][i].set_ylim([-1, self.gridworld.size])

    def plot_samples(self):
        self.samples.cla()
        current_samples = np.zeros((len(self.agent.samples), 2))
        for i, s in enumerate(self.agent.samples):
            p = self.gridworld.grid_position_from_state(s)
            current_samples[i][0] = p.x
            current_samples[i][1] = p.y
        self.samples.scatter(current_samples[:, 0], current_samples[:, 1])