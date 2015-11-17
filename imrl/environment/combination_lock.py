"""Combination lock environment."""

# System
import random
from itertools import islice

# First party
from imrl.environment.environment import Environment
from imrl.agent.option.option import Subgoal


class CombinationLock(Environment):
    """
    A combination lock has N tumblers of length L.  Each tumbler's sequence of potentially M numbers must be satisfied in order to unlock the combination lock.  The lock's solution and the lock's current internal position are simply represented as a list of integers.
    """

    def __init__(self, n_tumblers, l_tumbler_length, m_actions, solution, failure_rate):
        self.n_tumblers = n_tumblers
        self.l_tumbler_length = l_tumbler_length
        self.m_actions = m_actions
        self.solution = solution
        self.failure_rate = failure_rate
        self.actions = []

    def reset(self):
        """Reset the combination lock to initial state and return it."""
        self.actions = []
        return self

    def num_states(self):
        """Total number of possible states."""
        return self.n_tumblers * self.l_tumbler_length

    def reward_vector(self):
        return self.num_states() - 1

    def reward(self, state):
        # TODO
        return 0.0

    def initial_state(self):
        """Always start the lock with no tumblers set."""
        return 0

    def create_subgoals(self):
        subgoal_states = list(islice(range(1, self.num_states() + 1), 0, None, self.l_tumbler_length))
        return list(map(Subgoal, subgoal_states))

    def position_from_state(self, state):
        """Given a state, return the internal position that corresponds to this."""
        return self.solution[:state]

    def is_terminal(self):
        """It's terminal if the the combination lock is unlocked or if the action taken is incorrect for the current tumbler."""
        is_terminal = False
        for x, y in zip(self.actions, self.solution):
            if x != y:
                is_terminal = True
                break
        if self.actions == self.solution:
            is_terminal = True
        return is_terminal

    def next_state(self, state, action):
        """Apply the given action and return the next state.  This should never be called if state is already terminal."""
        self.actions = self.actions + [action]
        potential_position = self.position_from_state(state) + [action]
        if random.random() > self.failure_rate and (not self.is_terminal() or potential_position == self.solution):
            return state + 1
        else:
            return state
