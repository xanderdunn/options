"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

# Third party
import numpy as np


class FunctionApproximator:
    """Abstract function approximator class. Evaluates """

    def __init__(self, num_features, num_actions):
        self.num_features = num_features
        self.num_actions = num_actions
        self.size = num_features * num_actions

    def evaluate(self, s):
        raise NotImplementedError("Should evaluate a state.")

    def evaluate_state_action(self, s, a):
        raise NotImplementedError("Should evaluate a state-action pair.")

    def augment_with_action(self, fv, action):
        sa_fv = [0] * self.num_actions
        sa_fv[self.num_features * action:self.num_features*(action + 1)] = fv.tolist()
        return np.asarray(sa_fv)
