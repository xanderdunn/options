import logging


class Experiment2:

    def __init__(self, agent, environment, interval_length, steps, viz_steps):
        self.interval_length = interval_length
        self.max_steps = steps
        self.step = 0
        self.interval = 0
        self.agent = agent
        self.environment = environment
        self.viz_steps = viz_steps

    def run(self):
        self.s = self.environment.initial_state()
        while self.step < self.max_steps:
            self.run_interval()
            self.agent.plan()


    def run_interval(self):
        if self.interval >= 0:
            self.agent.policy = self.agent.vi_policy
        logging.info('Starting interval {}'.format(self.interval))
        while True:
            a = self.agent.choose_action(self.s)
            s_prime = self.environment.next_state(self.s, a)
            self.agent.update(self.s, a, s_prime)
            self.s = s_prime
            self.step += 1
            if self.step % (len(self.agent.samples) + 2) == 0:  # self.interval_length == 0:
                break
            if self.step % self.viz_steps == 0:
                if self.agent.viz:
                    self.agent.viz.update()
        self.interval += 1