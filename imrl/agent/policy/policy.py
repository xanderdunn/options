class Policy:

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, state):
        raise NotImplementedError("Should choose an action given the current state.")