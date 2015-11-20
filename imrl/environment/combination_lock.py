"""Combination lock environment."""

# System
import random
from itertools import islice

# Third party
import numpy as np

# First party
from imrl.environment.environment import Environment
from imrl.agent.option.option import Subgoal
from imrl.environment.gridworld import GridPosition


class CombinationLock(Environment):
    """A combination lock instance has N tumblers, each of which requires L correct actions to set.
    Actions are chosen from a set of size M.  The lock's solution is represented as a list of integers,
    which is the correct sequence of actions."""

    def __init__(self, n_tumblers, l_tumbler_length, num_actions, failure_rate, solution=None):
        super(CombinationLock, self).__init__(num_actions)
        self.n_tumblers = n_tumblers
        self.l_tumbler_length = l_tumbler_length
        self.size = l_tumbler_length
        if not solution:
            self.solution = [self.num_actions-1] * self.num_states()
            for i in range(self.num_actions-1):
                self.solution[i*self.l_tumbler_length:(i+1)*self.l_tumbler_length] = [i] * self.l_tumbler_length
        else:
            self.solution = solution
        self.failure_rate = failure_rate

    def num_states(self):
        """Total number of possible states."""
        return self.n_tumblers * self.l_tumbler_length

    def reward_vector(self):
        return self.num_states() - 1

    def grid_position_from_state(self, state):
        """Return a coordinate pair based on the given state vector."""
        return GridPosition(state % self.size, state // self.size)

    def reward(self, state):
        # TODO
        return 0.0

    def initial_state(self):
        """Always start the lock with no tumblers set."""
        return 0

    def create_subgoals(self):
        subgoal_states = list(range(self.l_tumbler_length - 1, self.num_states(), self.l_tumbler_length))
        return list(map(Subgoal, subgoal_states))

    def actions_from_state(self, state):
        """Given a state, return the list of actions that have been taken to get to this state."""
        return self.solution[:state]

    def is_terminal(self, actions):
        """Return true if the agent has made a wrong move."""
        is_terminal = False
        for x, y in zip(actions, self.solution):
            if x != y:
                is_terminal = True
                break
        if actions[:-1] == self.solution:
            is_terminal = True
        return is_terminal

    def next_state(self, state, action):
        """Apply the given action and return the next state.  This should never be called if state is already terminal."""
        potential_position = self.actions_from_state(state) + [action]
        if random.random() > self.failure_rate:
            if self.is_terminal(potential_position) or state == self.num_states() - 1:
                return self.initial_state()
            else:
                return state + 1
        else:
            return state
