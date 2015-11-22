import logging
import numpy as np

class Experiment2:

    def __init__(self, agent, environment, interval_length, steps, viz_steps):
        self.interval_length = interval_length
        self.max_steps = steps
        self.step = 0
        self.interval = 1
        self.agent = agent
        self.environment = environment
        self.viz_steps = viz_steps

    def run(self):
        self.s = self.environment.initial_state()
        while self.step < self.max_steps:
            self.run_interval()
            self.agent.plan()
            if self.step % self.viz_steps == 0:
                if self.agent.viz:
                    self.agent.viz.update()
                # self.print_models()

    def run_interval(self):
        if self.interval > 0:
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
        self.interval += 1

    def print_models(self):
        for o in self.agent.options.values():
            transitions = []
            returns = []
            for s in self.agent.samples:
                transitions.append((s, o.get_next_fv_from_state(s)))
                returns.append((s, self.agent.vi.get_value(self.agent.vi.theta, o, self.agent.fa.evaluate(s))))
                # returns.append((s, o.get_return_from_state(self.agent.intrinsic[o.id], s)))
            # print('Transitions:')
            # print('\t Option ' + str(o.id) + ': ' + str([(str(s) + ' --> ' + str(np.argmax(s_prime))) for (s, s_prime) in transitions]))
            # print('Returns:')
            print('\t Option ' + str(o.id) + ': ' + str([(str(s) + ' --> ' + str(np.argmax(rtn))) for (s, rtn) in returns]))
