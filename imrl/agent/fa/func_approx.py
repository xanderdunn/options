"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

import numpy as np

class FunctionApproximator:

    def __init__(self, input_dim, feature_dim):
        self.input_dim = input_dim
        self.feature_dim - feature_dim

