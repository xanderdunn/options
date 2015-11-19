import logging


class Experiment2:

    def __init__(self, agent, environment, interval_length, steps):
        self.interval_length = interval_length
        self.max_steps = steps
        self.step = 0
        self.interval = 1
        self.agent = agent
        self.environment = environment

    def run(self):
        self.s = self.environment.initial_state()
        while self.step < self.max_steps:
            self.run_interval()
            self.agent.plan()
            if self.agent.viz and self.step % 1000 == 0:
                self.agent.viz.update()

    def run_interval(self):
        if self.interval > 0:
            self.agent.policy = self.agent.vi_policy
        logging.info('Starting interval {}'.format(self.interval))
        while True:
            a = self.agent.policy.choose_action(self.s)
            s_prime = self.environment.next_state(self.s, a)
            self.agent.update(self.s, a, s_prime)
            self.s = s_prime
            self.step += 1
            if self.step % self.interval_length == 0:
                break
        self.interval += 1
