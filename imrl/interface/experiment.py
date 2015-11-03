"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging

# IMRL
from imrl.agent import agent_gridworld
from imrl.environment import gridworld


def generate_episode_id(num_episodes):
    """Episode ID"""
    for i in range(num_episodes):
        yield i


def generate_step_id(step):
    '''Step ID'''
    yield step + 1


def take_step(state, environment):
    '''Given the current state, choose an action to take and return the new environment state.'''
    action = agent_gridworld.decide_action(state)
    return gridworld.take_action(state, action, environment)


def generate_states(state, environment):
    '''A generator that represents all states that will be visited in an episode.'''
    if not state.is_terminal:
        yield from generate_states(take_step(state, environment), environment)
    else:
        yield state


def start(num_episodes, environment):
    '''Kick off the execution of an experiment.'''
    for episode_id in generate_episode_id(num_episodes):
        logging.info('Starting episode {}'.format(episode_id))
        for _ in generate_states(gridworld.State(gridworld.Position(0, 0), False, None), environment):
            logging.info('Terminated')
