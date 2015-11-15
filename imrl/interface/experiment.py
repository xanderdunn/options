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


ExperimentDescriptor = namedtuple('ExperimentDescriptor', ('num_value_iterations',           # Number of value iteration passes to run
                                                           'value_iterations_interval',      # Do value iteration updates every n intervals
                                                           'num_episodes'))                  # Number of episodes to execute
EpisodeData = namedtuple('EpisodeData', ('agent', 'results', 'episode_id'))
StepData = namedtuple('StepData', ('state',    # The environment's current state
                                   'action',   # The action the agent took that got the environment into this state
                                   'step_id',  # The time step of this state
                                   'agent'))   # The current agent after having received reward and updated for this state


def run_step(step_data, environment):
    """Given the current state, choose an action to take and return the new environment state."""
    state = step_data.state
    agent = step_data.agent
    assert not environment.is_terminal(step_data.state)
    action = agent.policy.choose_action(state)
    state_prime = environment.next_state(state, action)
    agent.update(state, action, state_prime)
    return StepData(state_prime, action, step_data.step_id + 1, agent)


def generate_state(agent, environment):
    """Run a single episode to termination."""
    initial_step_data = StepData(environment.initial_state(), None, 0, agent)  # Initial state with step_id 0
    return iterate(lambda step_data: run_step(step_data, environment), initial_step_data)


def run_episode(episode_id, initial_agent, environment, num_value_iterations, value_iterations_interval):
    """Run through a single episode to termination.  Returns the results for that episode."""
    logging.info('Starting episode {}'.format(episode_id))
    if episode_id % value_iterations_interval == 0:
        print('Planning...')
        initial_agent.plan()
        if initial_agent.viz:
            initial_agent.viz.update()
    for step_data in generate_state(initial_agent, environment):
        if environment.is_terminal(step_data.state):
            agent = step_data.agent
            agent.terminal_update(step_data.state, step_data.action, step_data.state)
            results = m(episode_id=episode_id, steps=step_data.step_id)
            return EpisodeData(agent, results, episode_id)


def episode_results_generator(agent, environment, experiment_description):
    """Execute episodes and yield the results for each episode."""
    return iterate_results(lambda episode_data: run_episode(episode_data.episode_id + 1, episode_data.agent,
                                                            environment, experiment_description.num_value_iterations,
                                                            experiment_description.value_iterations_interval), EpisodeData(agent, None, 0))


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(experiment_description, agent, environment, results_descriptor):
    """Kick off the execution of an experiment."""
    initialize_results(results_descriptor)
    episode_results = islice(episode_results_generator(agent, environment, experiment_description), experiment_description.num_episodes)
    results_episode_chunks = chunked(episode_results, results_descriptor.interval)
    for chunk in results_episode_chunks:
        results = [episode_data.results for episode_data in chunk]
        write_results(merge_results(results), results_descriptor)
