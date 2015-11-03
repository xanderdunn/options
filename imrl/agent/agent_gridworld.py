"""Managing data structures and algorithms used by an IMRL agent."""

# System
import random

# IMRL
from imrl.environment.gridworld import Action


def decide_action(state):
    '''Receive the environment's latest state and return an action to take.'''
    return random.choice(list(Action))
