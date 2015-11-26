"""Chemistry lab environment."""

# System
import collections

# Third party
import numpy as np

# First party
from imrl.environment.environment import Environment
from imrl.agent.option.option import Subgoal


class SubgoalNode(Subgoal):
    """A subgoal that contains a list connecting subgoals.  Set root = True if this is the root of the graph."""
    def __init__(self, state, radius=0.1):
        super(SubgoalNode, self).__init__(state, radius)
        self.connections = []
        self.root = False


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
        # TODO: This will explode when called
        return self.num_states() - 1

    def reward(self, state):
        return 0.0

    def initial_state(self):
        """The starting position is 0 at every dimension except the first, which starts at 0.5."""
        position = [max(np.random.normal(0, self.move_sd), 0.0) for _ in range(self.num_actions)]
        position[0] = 0.5
        return position

    @staticmethod
    def create_subgoals():
        """Create connected subgoals as described on the whiteboard and return the root node."""
        a = SubgoalNode([0.5, 0.25])
        a.root = True
        b = SubgoalNode([0.25, 0.25])
        c = SubgoalNode([0.75, 0.25])

        d = SubgoalNode([0.25, 0.5])
        e = SubgoalNode([0.5, 0.5])
        f = SubgoalNode([0.75, 0.5])

        g = SubgoalNode([0.25, 0.75])
        h = SubgoalNode([0.5, 0.75])
        i = SubgoalNode([0.75, 0.75])

        a.connections += [b, c]
        b.connections.append(d)
        c.connections.append(f)

        d.connections.append(e)
        e.connections.append(h)

        h.connections += [g, i]
        return a


    @staticmethod
    def is_within_subgoals(state, subgoal1, subgoal2):
        """Given two subgoals, check if the agent's state is between them.  The subgoals are SubgoalNode objects and the state is an n-dimensional list of agent's position values."""
        within = True
        epsilon = subgoal1.radius
        for i in range(len(state)):  # Iterate all the state dimensions
            print(i)
            sorted_values = sorted([subgoal1.state[i], subgoal2.state[i]])
            print(sorted_values)
            print(epsilon)
            within = state[i] <= sorted_values[1] + epsilon and state[i] >= sorted_values[0] - epsilon
            print(state[i])
            if not within:
                break
        print(within)
        print('\n')
        return within

    def is_terminal(self, state, subgoal):
        """Return true if the agent made a wrong move.  Given a subgoal (usually the root subgoal), check if the agent is within some valid connection between subgoals."""
        is_terminal = True
        next_subgoals = collections.deque([subgoal])
        while (len(next_subgoals) > 0):
            print(len(next_subgoals))
            current_subgoal = next_subgoals.pop()
            for next_subgoal in current_subgoal.connections:
                print('Adding subgoals {}'.format(next_subgoal))
                next_subgoals.appendleft(next_subgoal)
            for connection in current_subgoal.connections:
                is_terminal = not ChemistryLab.is_within_subgoals(state, current_subgoal, connection)
                if not is_terminal:
                    print('Got is_terminal=False from subgoal {} and connection {} and state {}'.format(subgoal, connection, state))
                    break
            if not is_terminal:
                break
        print('is_terminal = {}'.format(is_terminal))
        print('subgoal.root = {}'.format(subgoal.root))
        if is_terminal and subgoal.root:
            # Create a fake initial state subgoal just to see if it's in the corridor between initial state and the root subgoal
            fake_initial_state_subgoal = SubgoalNode(self.initial_state())
            is_terminal = not ChemistryLab.is_within_subgoals(state, subgoal, fake_initial_state_subgoal)
            if not is_terminal:
                print('Got is_terminal=False from subgoal {} and connection {} and state {}'.format(subgoal, fake_initial_state_subgoal, state))
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
