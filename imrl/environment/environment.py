"""Abstract class for environment."""


class Environment(object):

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def reward(self, state):
        raise NotImplementedError("Should return reward for transitioning into given state.")

    def initial_state(self):
        raise NotImplementedError("Should return initial state.")

    def next_state(self, state, action):
        raise NotImplementedError("Should return next state for given state and action.")

    def is_terminal(self, state):
        raise NotImplementedError("Should return true if given state is terminal.")

    def create_subgoals(self):
        raise NotImplementedError("Should return a set of subgoal states/regions.")
