"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging


def episode_ids(num_episodes):
    '''Return a generator for all episode IDs.'''
    return (i for i in range(num_episodes))


def generate_step_id(step):
    '''Step ID'''
    yield step + 1


def run_step(state, agent, environment):
    '''Given the current state, choose an action to take and return the new environment state.'''
    action = agent.decide_action(agent.policy, state, environment.num_actions)
    return environment.take_action(state, action, environment)


def generate_states(state, agent, environment):
    '''A generator that represents all states that will be visited in an episode.'''
    if not state.is_terminal:
        yield from generate_states(run_step(state, agent, environment), agent, environment)
    else:
        yield state


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(num_episodes, agent, environment):
    '''Kick off the execution of an experiment.'''
    for episode_id in episode_ids(num_episodes):
        logging.info('Starting episode {}'.format(episode_id))
        for _ in generate_states(environment.initial_state(), agent, environment):
            logging.info('Terminated')
