"""Stochastic policy implementation."""

import random


def policy_random(state, num_actions):
    """A policy where the an action is randmoly chosen."""
    return random.randint(0, num_actions - 1)


class Policy:

    def __init__(self, id):
        self.id = id
