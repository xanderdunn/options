"""Managing data structures and algorithms used by an IMRL agent."""

# System
from collections import namedtuple


Agent = namedtuple('Agent', ('policy', 'decide_action'))


def decide_action(policy, state, num_actions):
    '''Receive the environment's latest state and return an action to take.'''
    return policy(state, num_actions)
