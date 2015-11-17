"""Manages data structures and methods necessary to learn and execute options in MDPs.
An option consists of a policy, a universal option model (UOM), and a termination function."""

# Third party
import numpy as np

# First party
from imrl.agent.option.uom import UOM


class Subgoal:

    def __init__(self, state, radius=0):
        self.state = state
        self.radius = radius

    def __eq__(self, other):
        if isinstance(self.state, int):
            return self.state == other.state
        else:
            return (self.state == other.state).all() and self.radius == other.radius

    def __repr__(self):
        return "Subgoal({})".format(self.state)

    def __str__(self):
        return "{}".format(self.state)


class Option:

    def __init__(self, id, fa, policy, alpha, gamma, subgoal):
        self.id = id
        self.fa = fa
        self.policy = policy
        self.uom = UOM(fa, alpha, gamma)
        self.subgoal = subgoal

    def get_next_fv(self, fv):
        """Get expected next feature vector given feature vector fv."""
        return np.dot(self.uom.m, fv)

    def get_return(self, r, fv):
        """Calculate the expected return for executing the option in the state corresponding to the feature vector fv
        given the reward function r."""
        return np.dot(r.T, np.dot(self.uom.u, fv))

    def is_terminal(self, fv):
        """Returns true if the option terminates in the given feature vector fv."""
        return True
