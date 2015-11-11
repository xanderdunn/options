"""Test the environment state function approximators."""

# TODO: Test That the u and m matrices at the end of one episode are the same at the beginning of the next episode

# TODO: Test that the theta matrix is different at the beginning of each episode

# TODO: Test that the value iteration theta converges in the discrete tabular 3x3 gridworld case to:
# [[ 0.99202794]
# [ 0.99401498]
# [ 0.996006  ]
# [ 0.99401498]
# [ 0.996006  ]
# [ 0.998001  ]
# [ 0.996006  ]
# [ 0.998001  ]
# [ 1.        ]]

# System
import os

# First party
from imrl.interface.experiment import start, ExperimentDescriptor
from imrl.environment.gridworld import gridworld_discrete
from imrl.agent.agent import agent_random_tabular
from imrl.utils.results_writer import ResultsDescriptor


def test_discrete_gridworld_experiment():
    """Test the learning on an n*n tabular, discrete gridworld."""
    gridworld_size = 3
    num_episodes = 100
    environment = gridworld_discrete(gridworld_size, 0.0)
    experiment_description = ExperimentDescriptor(5, 5, num_episodes, None)
    agent = agent_random_tabular(gridworld_size * gridworld_size, environment.num_actions, 1.0, 1.0, 0.999)
    results_path = os.path.join(os.getcwd(), 'results.txt')
    results_descriptor = ResultsDescriptor(100, results_path, ['episode_id', 'steps'])
    start(experiment_description, agent, environment, results_descriptor)
