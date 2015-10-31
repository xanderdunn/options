"""Tabular function approximator. Implements 1-to-1 mapping from states to features."""

import numpy as np
from imrl.agent.fa.func_approx import FunctionApproximator


class TabularFA(FunctionApproximator):

    def __init__(self, n_states):
        super(TabularFA, self).__init__(n_states, n_states)
