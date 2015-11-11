# -*- coding: utf-8 -*-
"""Radial basis function approximator."""

# Set positions of Gaussians in a 2D grid.
# No state space that is zero for all RBFs
# Feature vector is the gaussian of all the rbfs
# Dimensions from 0 to 1
# Evenly tile all the dimensions
# Parameter d is how many RBFs I want across a single dimension
# You'll have d^n RBFs where n is the number of dimensions.  d**2 for a 2D gridworld

# Third party
from scipy.linalg import norm
from scipy import random, zeros, exp

# This class was adapted from Thomas Rückstieß at http://www.rueckstiess.net/research/snippets/show/72d2363e


class RBF:

    def __init__(self, indim, numCenters):
        self.indim = indim
        self.numCenters = numCenters
        # TODO: Use linspace from 0 to 1 across each dimension
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def calculate_activations(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
