"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

# System
import math

# First party
from imrl.utils.linear_algebra import sparse_one_hot_vector


def tabular_function_approximator(state, num_states):
    """Given a state represented as an integer, return the \"one-hot vector\" that is all zeros and a 1 at the position representing the occupied state.  Because these discrete states are exactly represented, this is not truly an approximation.  It's assumed that the states start at 0."""
    state_value = state.position.x + math.sqrt(num_states) * state.position.y
    assert isinstance(num_states, int), 'The discrete state space must be an int'
    assert state_value <= num_states and state_value >= 0, 'Given state {} with num_states {} is not possible'.format(state_value, num_states)
    return sparse_one_hot_vector(num_states, state_value)
