"""Chemistry lab environment."""

# Third party
import numpy as np

# First party
from imrl.environment.environment import Environment
from imrl.agent.option.option import Subgoal

# Subgoals are evenly spaced points with the same radius, not necessarily the width of the cooridor.
# Corridors between these points have a width outside of which the agent is sent back to the start state.
# Might represent it as a graph.  Edges are corridors between them.


class ChemistryLab(Environment):
    """
    A chemistry lab.  Actions are represented as arrays of length n, where n is the number of parameters in the lab that the agent can control.  In the simplest case, an action array is [T, A] where T is temperature and A is adding some chemical. T can be both increased and decreased.  All other parameters can only be increased.  The reward_radius is used both as the radius of subgoals and as half the width of the cooridors between subgoals.
    """

    def __init__(self, move_mean=0.05, move_sd=0.01, reward_radius=0.1, num_actions=2):
        super(ChemistryLab, self).__init__(num_actions)
        self.move_mean = move_mean
        self.move_sd = move_sd
        self.reward_radius = reward_radius

    def reward_vector(self):
        # TODO
        return self.num_states() - 1

    def reward(self, state):
        return 0.0

    def initial_state(self):
        """The starting position is 0 at every dimension except the first, which starts at 0.5."""
        position = [max(np.random.normal(0, self.move_sd), 0.0) for _ in range(self.num_actions)]
        position = position[:, 0] = 0.5
        return position

    def create_subgoals(self):
        # TODO
        subgoal_states = list(islice(range(1, self.num_states() + 1), 0, None, self.l_tumbler_length))
        return list(map(Subgoal, subgoal_states))

    def is_terminal(self, state):
        """Return true if the agent has made a wrong move."""
        # TODO: Test that the potential position is within at least one edge to a conencted Subgoal
        # TODO: Test if the state is at the final subgoal
        is_terminal = False
        for x, y in zip(actions, self.solution):
            if x != y:
                is_terminal = True
                break
        if actions[:-1] == self.solution:
            is_terminal = True
        return is_terminal

    def next_state(self, state, action):
        """Apply the given action and return the next state.  This should never be called if state is already terminal.  The action being altered is that action's integer value starting at 0."""
        noise = np.random.normal(0, self.move_sd)
        move = np.random.normal(self.move_mean, self.move_sd)
        potential_state = np.full((1, self.num_actions), noise)
        potential_state[:, action] = move
        potential_state = np.add(state, potential_state)

        if self.is_terminal(potential_state):
            return self.initial_state()
        else:
            return potential_state
