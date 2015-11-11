"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

# First party
from imrl.utils.linear_algebra import sparse_one_hot_vector


def tabular_function_approximator(state, num_states):
    """Given a state represented as an integer, return the \"one-hot vector\" that is all zeros and a 1 at the position representing the occupied state.  Because these discrete states are exactly represented, this is not truly an approximation.  It's assumed that the states start at 0."""
    assert isinstance(num_states, int), 'The discrete state space must be an int'
    assert isinstance(state, int), 'Received state {}, but the state provided to the tabular function approximator must be an int'.format(state)
    assert state <= num_states and state >= 0, 'Given state {} with num_states {} is not possible'.format(state, num_states)
    return sparse_one_hot_vector(num_states, state)
