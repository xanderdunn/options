"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging
from collections import namedtuple

# Third party
from more_itertools import iterate


StepData = namedtuple('StepData', ('state', 'step_id'))


def episode_id_generator(num_episodes):
    '''Return a generator for all episode IDs.'''
    return (i for i in range(num_episodes))


def run_step(step_data, agent, environment):
    '''Given the current state, choose an action to take and return the new environment state.'''
    state = step_data.state
    if state.is_terminal:
        logging.info('Terminated')
    action = agent.decide_action(agent.policy, state, environment.num_actions)
    state = environment.take_action(state, action, environment)
    return StepData(state, step_data.step_id + 1)


def generate_state(agent, environment):
    '''Run a single episode to termination.'''
    initial_step_data = StepData(environment.initial_state(), 0)
    return iterate(lambda state: run_step(state, agent, environment), initial_step_data)


def run_episode(agent, environment):
    '''Run through a single episode to termination.'''
    for step_data in generate_state(agent, environment):
        if step_data.state.is_terminal:
            logging.info('Terminated in state {}'.format(step_data))
            break
        else:
            logging.debug('Visited state {}'.format(step_data))


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(num_episodes, agent, environment, logging_interval):
    '''Kick off the execution of an experiment.'''
    for episode_id in episode_id_generator(num_episodes):
        logging.info('Starting episode {}'.format(episode_id))
        run_episode(agent, environment)
