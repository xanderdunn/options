"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

import numpy as np
from scipy.sparse import csc_matrix


def tabular_function_approximator(state, num_states):
    """Given a state represented as an integer, return the \"one-hot vector\" that is all zeros and a 1 at the position representing the occupied state.  Because these discrete states are exactly represented, this is not truly an approximation.  It's assumed that the states start at 0."""
    assert isinstance(num_states, int), 'The discrete state space must be an integer'
    assert isinstance(state, int), 'The provided state must be a float'
    assert state <= num_states and state >= 0, 'Given state {} with num_states {} is not possible'.format(state, num_states)
    zero_vector = np.zeros((num_states, 1))
    zero_vector[state, 0] = 1.0
    return csc_matrix(zero_vector)


class FunctionApproximator:

    def __init__(self, input_dim, feature_dim):
        self.input_dim = input_dim
        self.feature_dim - feature_dim
