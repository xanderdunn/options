# Third party
import numpy as np
import matplotlib.pyplot as plt

# First party
from imrl.agent.agent_viz import AgentViz
# from imrl.environment.gridworld import Action

class AgentVizDisc(AgentViz):

    def __init__(self, agent, num_options, environment):
        self.environment = environment
        super(AgentVizDisc, self).__init__(agent, num_options)

    def make_grid_samples(self):
        self.grid_samples = np.zeros((self.environment.num_states(), 2))
        self.state_samples = []
        for i in range(self.environment.num_states()):
            p = self.environment.grid_position_from_state(i)
            self.grid_samples[i][0] = p.x
            self.grid_samples[i][1] = p.y
            self.state_samples.append(i)

    def update_plot_limits(self):
        for i in range(self.num_options + 1):
            self.subplots['policy'][i].set_xlim([-1, self.environment.width])
            self.subplots['policy'][i].set_ylim([-1, self.environment.height])

    def plot_samples(self):
        self.samples.cla()
        current_samples = np.zeros((len(self.agent.samples), 2))
        for i, s in enumerate(self.agent.samples):
            p = self.environment.grid_position_from_state(s)
            current_samples[i][0] = p.x
            current_samples[i][1] = p.y
        self.samples.scatter(current_samples[:, 0], current_samples[:, 1])