"""Universal option model (UOM). Manages learning and inference for UOMs."""

# Third party
import numpy as np


class UOM:

    def __init__(self, fa, eta, gamma):
        self.m = np.zeros((fa.num_features, fa.num_features))
        self.u = np.zeros((fa.num_features, fa.num_features))
        self.eta = eta
        self.gamma = gamma

    def converged(m, m_prime, u, u_prime, epsilon):
        """Check if the model has converged sufficiently."""
        return m - m_prime < epsilon

    def update_m(self, fv, fv_prime, tau):
        """Update M based on the previous feature vector fv and the next feature vector fv_prime."""
        assert fv.shape == fv_prime.shape, 'The feature vectors must be the same shape.'
        m_prime = self.m + self.eta * ((self.gamma ** tau) * fv_prime - np.dot(self.m, fv)) * fv.T
        assert m_prime.shape == self.m.shape, 'The updated M\' must have the same shape as the previous M.'
        self.m = m_prime
        return m_prime

    def update_u(self, fv):
        """Given the current matrix U and the previous feature vector fv, return the updated matrix U."""
        u_prime = self.u + self.eta * (fv - np.dot(self.u, fv)) * fv.T
        assert u_prime.shape == self.u.shape, 'The updated U\' must have the same shape as the previous U.'
        self.u = u_prime
        return u_prime
