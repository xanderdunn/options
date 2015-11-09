"""Universal option model (UOM). Manages learning and inference for UOMs."""

# System
from collections import namedtuple

# Third party
import numpy as np


UOMDescriptor = namedtuple('UOMDescriptor', ('eta',      # learning rate
                                             'gamma',    # Discount factor
                                             'tau',      # time taken for the option to execute
                                             'converged',
                                             'epsilon'))
UOM = namedtuple('UOM', ('descriptor', 'm', 'u'))


def uom_primitive(fv_size):
    descriptor = UOMDescriptor(1.0, 0.999, 1.0, converged, 0.001)
    return UOM(descriptor, initial_m(fv_size), initial_u(fv_size))


def initial_m(fv_size):
    """The initial M matrix is all zeros."""
    return np.zeros((fv_size, fv_size))


def initial_u(fv_size):
    """The initial U matrix is all zeros."""
    return np.zeros((fv_size, fv_size))


def converged(m, m_prime, u, u_prime, epsilon):
    """Check if the model has converged sufficiently."""
    return (m - m_prime < epsilon) or False


def update_m(uom, fv, fv_prime):
    """Given the current matrix M, the previous feature vector fv, and the next feature vector fv_prime, return the updated matrix M."""
    eta = uom.descriptor.eta
    gamma = uom.descriptor.gamma
    tau = uom.descriptor.tau
    m = uom.m
    assert fv.shape == fv_prime.shape, 'The feature vectors must be the same shape.'
    m_prime = m + eta * ((gamma ** tau) * fv_prime - np.dot(m, fv)) * np.transpose(fv)
    assert m_prime.shape == m.shape, 'The updated matrix M\' must have the same shape as the previous matrix M.'
    return m_prime


def update_u(uom, fv):
    """Given the current matrix U and the previous feature vector fv, return the updated matrix U."""
    eta = uom.descriptor.eta
    u = uom.u
    u_prime = u + eta * (fv - np.dot(u, fv)) * np.transpose(fv)
    assert u_prime.shape == u.shape, 'The updated matrix U\' must have the same shape as the previous matrix U.'
    return u_prime
