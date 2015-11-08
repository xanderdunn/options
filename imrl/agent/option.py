"""Manages data structures and methods necessary to learn and execute hierarchical options in MDPs.  An Option consists of a policy, a UOM that represents the model, and a function to determine if the option is terminal."""

# System
from collections import namedtuple

# First Party
from imrl.agent.policy import policy_primitive
from imrl.agent.uom import uom_primitive

OptionDescriptor = namedtuple('OptionDescriptor', ('policy', 'is_terminal'))
Option = namedtuple('Option', ('descriptor', 'uom'))


def option_primitive(fv_size):
    """An option with primitive actions where it is always in the terminal state and always executes exactly the primitive action."""
    descriptor = OptionDescriptor(policy_primitive, is_terminal_always)
    return Option(descriptor, uom_primitive(fv_size))


def get_next_feature(fv):
    """Where fv is a feature vector."""
    return M * phi(fv)


def get_return(f, fv):
    """Take the reward function f and the feature vector fv and calculate the return for ???"""
    return f * U * phi(fv)


def is_terminal_always(fv):
    """Will eventually return 1 if you're in..."""
    return 1
