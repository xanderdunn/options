"""Stochastic policy implementation."""

# System
import random

# First party
import imrl.agent.value_iteration as value_iteration


def policy_random(agent, state, num_actions, reward_function):
    """A policy where the an action is randmoly chosen."""
    return random.randint(0, num_actions - 1)


def policy_value_iteration(agent, state, num_actions, reward_function):
    """Given reward function f, feature vector fv, universal option model matrixes u and m, and theta."""
    f = reward_function
    fv = agent.descriptor.feature_vector(state, agent.descriptor.num_states)
    theta = agent.computed_policy
    uoms = [option.uom for option in agent.options]
    values = [value_iteration.scalar(theta, f, uom, fv) for uom in uoms]
    max_value = max(values)
    max_value_actions = [i for i, x in enumerate(values) if x == max_value]
    return random.choice(max_value_actions)
