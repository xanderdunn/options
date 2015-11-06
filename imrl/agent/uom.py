"""Universal option model (UOM). Manages learning and inference for UOMs."""

# System
from scipy.sparse import csc_matrix
from collections import namedtuple

# Third party
import numpy as np


UOM = namedtuple('UOM', ['eta',      # learning rate
                         'gamma',    # Discount factor
                         'tau',      # time taken for the option to execute
                         'initial_m',
                         'initial_u',
                         'converged', 
                         'epsilon'])


def uom_primitive(fv_size):
    return UOM(1.0, 0.999, 1.0, initial_m(fv_size), initial_u(fv_size), converged, 0.001)


def initial_m(fv_size):
    """The initial M matrix is all zeros."""
    return csc_matrix((fv_size, fv_size), dtype=np.float64)


def initial_u(fv_size):
    """The initial U matrix is all zeros."""
    return csc_matrix((fv_size, fv_size), dtype=np.float64)


def converged(m, m_prime, u, u_prime, epsilon):
    """Check if the model has converged sufficiently."""
    return (m - m_prime < epsilon) or false


def update_m(uom, m, fv, fv_prime):
    """Given the current matrix M, the previous feature vector fv, and the next feature vector fv_prime, return the updated matrix M."""
    eta = uom.eta
    gamma = uom.gamma
    tau = uom.tau
    assert fv.shape == fv_prime.shape, 'The feature vectors must be the same shape.'
    m_prime = m + eta * ((gamma ** tau) * fv_prime - m * fv) * np.transpose(fv)
    assert m_prime.shape == m.shape, 'The updated matrix M\' must have the same shape as the previous matrix M.'
    return m_prime


def update_u(uom, u, fv):
    """Given the current matrix U and the previous feature vector fv, return the updated matrix U."""
    eta = uom.eta
    u_prime = u + eta * (fv - u * fv) * np.transpose(fv)
    assert u_prime.shape == u.shape, 'The updated matrix U\' must have the same shape as the previous matrix U.'
    return u_prime
