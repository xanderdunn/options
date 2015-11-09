"""Value iteration algorithm."""

# System
import random

# Third party
import numpy as np


def initial_theta(fv_size):
    """Sparse matrix that theta is initialized to."""
    return np.zeros((fv_size, 1))


def update_theta(theta, alpha, f, uoms, fvs):
    """Given the previous value iteration theta, learning rate alpha, reward function f, universal option model matrices u and m, feature vector sample set fv, and discount factor gamma, return the updated value iteration theta_prime."""
    # TODO: What is an elegant way to remove this mutable state?
    theta_prime = theta
    for fv in fvs:
        theta_prime = update_theta_single(theta_prime, alpha, f, uoms, fv)
    return theta_prime


def update_theta_single(theta, alpha, f, uoms, fv):
    """This is called by update_theta for a single feature vector fv."""
    max_value = max([np.dot(f.T, np.dot(uom.u, fv)) + uom.descriptor.gamma * np.dot(np.dot(uom.m, fv).T, theta) for uom in uoms])
    update_value = (alpha * (max_value - np.dot(fv.T, theta))) * fv
    return theta + update_value


def policy(f, uoms, fv, theta):
    """Given reward function f, feature vector fv, universal option model matrixes u and m, and theta."""
    values = [np.transpose(f) * uom.u * fv + uom.descriptor.gamma * np.transpose(uom.m * fv) * theta for uom in uoms]
    max_value = max(values)
    max_value_actions = [i for i, x in enumerate(values) if x == max_value]
    return random.choice(max_value_actions)
