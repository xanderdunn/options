"""Stochastic policy implementation."""

import random


def policy_random(state, num_actions):
    """A policy where the an action is randmoly chosen."""
    return random.randint(0, num_actions - 1)


def policy_value_iteration(pi, fv):
    """Given a policy pi calculated via value iteration, return the corresponding action."""
    return pi * fv
