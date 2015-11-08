"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging
from collections import namedtuple
from itertools import islice

# Third party
from more_itertools import iterate, chunked
from pyrsistent import m

# First party
from imrl.utils.results_writer import write_results, merge_results, initialize_results
from imrl.utils.iterators import iterate_results
from imrl.environment.gridworld import Position


EpisodeData = namedtuple('EpisodeData', ('agent', 'results', 'episode_id'))
StepData = namedtuple('StepData', ('state', 'step_id', 'agent'))


def episode_id_generator(num_episodes):
    """Return a generator for all episode IDs."""
    return (i for i in range(num_episodes))


def run_step(step_data, environment):
    """Given the current state, choose an action to take and return the new environment state."""
    state = step_data.state
    agent = step_data.agent
    assert not step_data.state.is_terminal
    action = agent.descriptor.decide_action(agent.descriptor.policy, state, environment.num_actions)
    state_prime = environment.take_action(state, environment.size, action, environment)
    agent_prime = agent.descriptor.update(agent, action, state, state_prime)
    return StepData(state_prime, step_data.step_id + 1, agent_prime)


def generate_state(agent, environment):
    """Run a single episode to termination."""
    initial_step_data = StepData(environment.initial_state(), 0, agent)  # Initial state with step_id 0
    return iterate(lambda step_data: run_step(step_data, environment), initial_step_data)


def run_episode(episode_id, agent, environment):
    """Run through a single episode to termination.  Returns the results for that episode."""
    logging.info('Starting episode {}'.format(episode_id))
    for step_data in generate_state(agent, environment):
        if step_data.state.is_terminal:
            results = m(episode_id=episode_id, steps=step_data.step_id)
            assert step_data.state.position == Position(environment.size - 1, environment.size - 1)
            assert step_data.state.value == step_data.state.position.x + step_data.state.position.y * environment.size
            return EpisodeData(step_data.agent, results, episode_id)
        else:
            logging.debug(step_data)


def episode_results_generator(agent, environment):
    """Execute episodes and yield the results for each episode."""
    return iterate_results(lambda episode_data: run_episode(episode_data.episode_id + 1, episode_data.agent, environment), EpisodeData(agent, None, 0))


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(num_episodes, agent, environment, results_descriptor):
    """Kick off the execution of an experiment."""
    initialize_results(results_descriptor)
    episode_results = islice(episode_results_generator(agent, environment), num_episodes)
    results_episode_chunks = chunked(episode_results, results_descriptor.interval)
    for chunk in results_episode_chunks:
        results = [episode_data.results for episode_data in chunk]
        write_results(merge_results(results), results_descriptor)
