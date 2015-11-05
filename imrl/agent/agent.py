"""Managing data structures and algorithms used by an IMRL agent."""

# System
import random
from collections import namedtuple


RandomAgent = namedtuple('RandomAgent', ('policy', 'decide_action'))


def policy_random(state, num_actions):
    '''A policy where the an action is randmoly chosen.'''
    return random.randint(0, num_actions - 1)


def decide_action(policy, state, num_actions):
    '''Receive the environment's latest state and return an action to take.'''
    return policy(state, num_actions)


class Agent:

    def __init__(self, policy):
        self.policy = policy
