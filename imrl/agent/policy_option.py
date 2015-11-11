"""Stochastic policy implementation."""


def policy_primitive(action, fv, num_actions):
    """The option policy for the primitive action situation is to simply execute the action this option is associated with."""
    return action
