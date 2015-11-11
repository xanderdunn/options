"""Controller to execute experiments. Provides agent-environment interface functionality."""

# System
import logging
from collections import namedtuple
from itertools import islice

# Third party
from more_itertools import iterate, chunked
from pyrsistent import m
from cytoolz import last

# First party
from imrl.utils.results_writer import write_results, merge_results, initialize_results
from imrl.utils.iterators import iterate_results
from imrl.environment.gridworld import Position
from imrl.agent.agent import Agent
from imrl.agent.policy_agent import policy_value_iteration
from imrl.agent.value_iteration import initial_theta


ExperimentDescriptor = namedtuple('ExperimentDescriptor', ('num_value_iterations',           # Number of value iteration passes to run
                                                           'value_iterations_interval',      # Do value iteration updates every n intervals
                                                           'num_episodes',                   # Number of episodes to execute
                                                           'value_iteration_policy_start'))  # Starting at this episode, the agent will be execute value iteration policy
EpisodeData = namedtuple('EpisodeData', ('agent', 'results', 'episode_id'))
StepData = namedtuple('StepData', ('state',    # The environment's current state
                                   'action',   # The action the agent took that got the environment into this state
                                   'step_id',  # The time step of this state
                                   'agent'))   # The current agent after having received reward and updated for this state


def run_step(step_data, environment):
    """Given the current state, choose an action to take and return the new environment state."""
    state = step_data.state
    agent = step_data.agent
    assert not step_data.state.is_terminal
    action = agent.descriptor.decide_action(agent, state, environment.num_actions, environment.reward_vector)
    state_prime = environment.take_action(state, environment.size, action, environment)
    agent_prime = agent.descriptor.update(agent, action, state, state_prime, 1.0)
    return StepData(state_prime, action, step_data.step_id + 1, agent_prime)


def generate_state(agent, environment):
    """Run a single episode to termination."""
    initial_step_data = StepData(environment.initial_state(), None, 0, agent)  # Initial state with step_id 0
    return iterate(lambda step_data: run_step(step_data, environment), initial_step_data)


def run_value_iteration(episode_id, agent, environment, num_value_iterations, value_iterations_interval):
    """Execute value iteration and return an updated Agent with the computed_policy."""
    if episode_id % value_iterations_interval == 0:
        uoms = [option.uom for option in agent.options]
        value_iteration_generator = iterate_results(lambda computed_policy_theta: agent.descriptor.compute_policy(computed_policy_theta, agent.descriptor.learning_rate, environment.reward_vector, uoms, environment.exhaustive_states), initial_theta(len(environment.exhaustive_states)))
        computed_policy = last(islice(value_iteration_generator, num_value_iterations))
        # TODO: I shouldn't be creating a new Agent object here, it should be in some API in the agent module
        return Agent(agent.descriptor, agent.options, computed_policy)
    else:
        return agent


def run_episode(episode_id, initial_agent, environment, num_value_iterations, value_iterations_interval, value_iteration_policy_start):
    """Run through a single episode to termination.  Returns the results for that episode."""
    logging.info('Starting episode {}'.format(episode_id))
    value_iterated_agent = run_value_iteration(episode_id, initial_agent, environment, num_value_iterations, value_iterations_interval)
    if episode_id == value_iteration_policy_start:
        value_iterated_agent = value_iterated_agent.descriptor.switch_policy(value_iterated_agent, policy_value_iteration)
    for step_data in generate_state(value_iterated_agent, environment):
        if step_data.state.is_terminal:
            agent = step_data.agent
            terminal_agent = agent.descriptor.terminal_update(agent, step_data.action, step_data.state)
            results = m(episode_id=episode_id, steps=step_data.step_id)
            assert step_data.state.position == Position(environment.size - 1, environment.size - 1)
            return EpisodeData(terminal_agent, results, episode_id)


def episode_results_generator(agent, environment, experiment_description):
    """Execute episodes and yield the results for each episode."""
    return iterate_results(lambda episode_data: run_episode(episode_data.episode_id + 1, episode_data.agent, environment, experiment_description.num_value_iterations, experiment_description.value_iterations_interval, experiment_description.value_iteration_policy_start), EpisodeData(agent, None, 0))


# Think about using tee and chain, or izip to generate the step_id along with the episode_id
def start(experiment_description, agent, environment, results_descriptor):
    """Kick off the execution of an experiment."""
    initialize_results(results_descriptor)
    episode_results = islice(episode_results_generator(agent, environment, experiment_description), experiment_description.num_episodes)
    results_episode_chunks = chunked(episode_results, results_descriptor.interval)
    for chunk in results_episode_chunks:
        results = [episode_data.results for episode_data in chunk]
        write_results(merge_results(results), results_descriptor)
