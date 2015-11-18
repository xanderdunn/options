"""Tabular function approximator. Implements 1-to-1 mapping from states to features."""

# First party
from imrl.agent.fa.func_approx import FunctionApproximator
from imrl.utils.linear_algebra import one_hot_vector

class TabularFA(FunctionApproximator):

    def __init__(self, num_states, num_actions):
        super(TabularFA, self).__init__(num_states, num_actions)

    def evaluate(self, s):
        assert isinstance(s, int), 'The input sample must be an int'
        assert self.num_features >= s >= 0, \
            'Given state {} with num_states {} is not possible'.format(s, self.num_features)
        return one_hot_vector(self.num_features, s)

    def evaluate_state_action(self, s, a):
        """Create a one-hot vector for the given state-action pair."""
        assert isinstance(s, int), 'The input sample must be an int'
        assert self.num_features >= s >= 0, \
            'Given state {} with num_states {} is not possible'.format(s, self.num_features)
        return one_hot_vector(self.size, self.num_features * a + s)