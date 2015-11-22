"""Manages data structures and methods necessary to learn and execute options in MDPs.
An option consists of a policy, a universal option model (UOM), and a termination function."""

# Third party
import numpy as np

# First party


class Subgoal:

    def __init__(self, state, radius=0):
        self.state = state
        self.radius = radius

    def __eq__(self, other):
        if isinstance(self.state, int):
            return self.state == other.state
        else:
            return (self.state == other.state).all() and self.radius == other.radius

    def __repr__(self):
        return "Subgoal({})".format(self.state)

    def __str__(self):
        return "{}".format(self.state)


class Option:

    def __init__(self, id, fa, policy, eta, gamma, subgoal, num_actions):
        self.id = id
        self.fa = fa
        self.policy = policy
        self.num_actions = num_actions
        self.m = np.zeros((fa.num_features, fa.num_features))
        self.u = np.eye(fa.num_features, fa.num_features)
        self.eta = eta
        self.gamma = gamma
        if subgoal:
            self.subgoal = subgoal
            self.subgoal_fv = fa.evaluate(subgoal.state)
            # self.subgoal_fv_tolerance = np.linalg.norm(np.asarray(self.subgoal_fv - fa.evaluate(subgoal.state - subgoal.radius)), 2)

    def get_next_fv(self, fv):
        """Get expected next feature vector given feature vector fv."""
        return np.dot(self.m, fv)

    def get_return(self, r, fv):
        """Calculate the expected return for executing the option in the state corresponding to the feature vector fv
        given the reward function r."""
        return np.dot(r.T, np.dot(self.u, fv))

    def get_next_fv_from_state(self, s):
        """Get expected next feature vector given state s."""
        return self.get_next_fv(self.fa.evaluate(s))

    def get_return_from_state(self, r, s):
        """Calculate the expected return for executing the option in the given state given the reward function r."""
        return self.get_return(r, self.fa.evaluate(s))

    def can_init_from_state(self, s):
        """Return true if option is initializable from state s."""
        return self.can_init_from_fv(self.fa.evaluate(s))

    def can_init_from_fv(self, fv):
        """Return true if option is initializable from feature vector fv."""
        return self.id < self.num_actions or np.argmax(fv) <= self.subgoal.state  # TODO only works for combo lock

    def get_init_set(self):
        return [s for s in self.policy.vi.agent.samples if self.can_init_from_state(s)]

    def is_terminal(self, fv):
        """Returns true if the option terminates in the given feature vector."""
        if self.id < self.num_actions:
            return True
        if np.linalg.norm((fv - self.subgoal_fv), 2) <= 0.1:
            return True
        return False

    def is_terminal_in_state(self, state):
        """Returns true if the option terminates in the given state."""
        if self.id < self.num_actions:
            return True
        if isinstance(state, int):  # Just check set membership for discrete domains.
            return state == self.subgoal.state
        if np.linalg.norm(np.asarray(self.subgoal.state - state), 2) <= self.subgoal.radius:
            return True
        return False

    def update_m(self, fv, fv_prime, tau):
        """Update M based on the previous feature vector fv and the next feature vector fv_prime."""
        assert fv.shape == fv_prime.shape, 'The feature vectors must be the same shape.'
        m_prime = self.m + self.eta * np.dot((self.gamma ** tau) * fv_prime - np.dot(self.m, fv), fv.T)
        assert m_prime.shape == self.m.shape, 'The updated M\' must have the same shape as the previous M.'
        self.m = m_prime
        return m_prime

    def update_u(self, fv, fv_prime, terminal):
        """Given the current matrix U and the previous feature vector fv, return the updated matrix U."""
        successor_val = np.zeros((self.fa.num_features, 1)) if terminal else np.dot(self.u, fv_prime)
        delta = fv + self.gamma * successor_val - np.dot(self.u, fv)
        u_prime = self.u + self.eta * np.dot(delta, fv.T)
        assert u_prime.shape == self.u.shape, 'The updated U\' must have the same shape as the previous U.'
        self.u = u_prime
        return u_prime
