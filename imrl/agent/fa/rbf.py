"""Radial basis function approximator."""

import numpy as np

from imrl.agent.fa.func_approx import FunctionApproximator


class Rbf(FunctionApproximator):

    def __init__(self, input_dim, tiling):
        super(Rbf, self).__init__(input_dim, tiling**input_dim)
        self.tiling = tiling