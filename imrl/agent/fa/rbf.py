"""Radial basis function approximator."""

# Third party
import numpy as np
from itertools import product
from scipy.linalg import norm
import matplotlib.pyplot as plt

class RBF:

    def __init__(self, dim, resolution, min_val=0, max_val=1, eps=8):
        self.dim = dim
        self.num_centers = resolution**dim
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

        # Segment the space based on resolution
        segmentation = [np.linspace(min_val, max_val, resolution).tolist()]*dim

        # Generate centers from Cartiesian product of segmentation lists
        centers = product(*segmentation)
        self.centers = [np.asarray(a) for a in centers]

    def _eval(self, c, x):
        """Evaluate an RBF kernel c at a given point x."""
        return np.exp(-self.eps * norm(c - x) ** 2)

    def get_features(self, s):
        """Get the feature vector for a given state s."""
        assert len(s) == self.dim
        for i in range(len(s)):
            assert self.min_val <= s[i] <= self.max_val
        fv = np.zeros((self.num_centers, 1))
        for ci, c in enumerate(self.centers):
            fv[ci] = self._eval(c, s)
        return fv


if __name__ == '__main__':
    rbfs = RBF(2, 5)
    fv = rbfs.get_features(np.asarray([0.5, 0.5]))
    print(fv)
    plt.plot(range(rbfs.num_centers), fv)
    plt.show()