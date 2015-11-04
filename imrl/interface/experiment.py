"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging

# IMRL
from imrl.agent import agent_gridworld
from imrl.environment import gridworld


def episode_ids(num_episodes):
    '''Return a generator for all episode IDs.'''
    return (i for i in range(num_episodes))


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


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(num_episodes, environment):
    '''Kick off the execution of an experiment.'''
    for episode_id in episode_ids(num_episodes):
        logging.info('Starting episode {}'.format(episode_id))
        for _ in generate_states(gridworld.State(gridworld.Position(0, 0), False, None), environment):
            logging.info('Terminated')
