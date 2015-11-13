"""Function approximator. Maps input vectors to feature vectors that approximate the input function."""

class FunctionApproximator(object):
    """Abstract function approximator class. Evaluates """

    def __init__(self, num_features):
        self.num_features = num_features

    def evaluate(self, s):
        raise NotImplementedError("Implement FA evaluation.")
