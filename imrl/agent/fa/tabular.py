"""Tabular function approximator. Implements 1-to-1 mapping from states to features."""

# System
import math

# First party
from imrl.agent.fa.func_approx import FunctionApproximator
from imrl.utils.linear_algebra import one_hot_vector
from imrl.environment.gridworld import GridPosition

class TabularFA(FunctionApproximator):

    def __init__(self, num_states):
        super(TabularFA, self).__init__(num_states)

    def evaluate(self, s):
        """Create a one-hot vector for the given state index."""
        assert isinstance(s, int), 'The input sample must be an int'
        assert self.num_features >= s >= 0, \
            'Given state {} with num_states {} is not possible'.format(s, self.num_features)
        return one_hot_vector(self.num_features, s)