"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging
from collections import namedtuple

# Third party
from more_itertools import iterate, chunked
from pyrsistent import m

# IMRL
from imrl.utils.results_writer import write_results, merge_results, initialize_results


StepData = namedtuple('StepData', ('state', 'step_id'))


def episode_id_generator(num_episodes):
    """Return a generator for all episode IDs."""
    return (i for i in range(num_episodes))


def run_step(step_data, agent, environment):
    """Given the current state, choose an action to take and return the new environment state."""
    state = step_data.state
    if state.is_terminal:
        logging.info('Terminated')
    # TODO: Separate this into sending state and getting an action
    action = agent.decide_action(agent.policy, state, environment.num_actions)
    state = environment.take_action(state, action, environment)
    return StepData(state, step_data.step_id + 1)


def generate_state(agent, environment):
    """Run a single episode to termination."""
    initial_step_data = StepData(environment.initial_state(), 0)
    return iterate(lambda state: run_step(state, agent, environment), initial_step_data)


def run_episode(episode_id, agent, environment):
    """Run through a single episode to termination.  Returns the results for that episode."""
    logging.info('Starting episode {}'.format(episode_id))
    for step_data in generate_state(agent, environment):
        if step_data.state.is_terminal:
            return m(episode_id=episode_id, steps=step_data.step_id)


def episode_results_generator(num_episodes, agent, environment):
    """Execute episodes and yield the results for each episode."""
    return (run_episode(episode_id, agent, environment) for episode_id in episode_id_generator(num_episodes))


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(num_episodes, agent, environment, results_descriptor):
    """Kick off the execution of an experiment."""
    initialize_results(results_descriptor)
    results_episode_chunks = chunked(episode_results_generator(num_episodes, agent, environment), results_descriptor.interval)
    for chunk in results_episode_chunks:
        write_results(merge_results(chunk), results_descriptor)
