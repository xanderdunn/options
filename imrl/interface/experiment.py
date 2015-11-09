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
    action = agent.descriptor.decide_action(agent.descriptor.policy, state, environment.num_actions)
    state_prime = environment.take_action(state, environment.size, action, environment)
    agent_prime = agent.descriptor.update(agent, action, state, state_prime)
    return StepData(state_prime, action, step_data.step_id + 1, agent_prime)


def generate_state(agent, environment):
    """Run a single episode to termination."""
    initial_step_data = StepData(environment.initial_state(), None, 0, agent)  # Initial state with step_id 0
    return iterate(lambda step_data: run_step(step_data, environment), initial_step_data)


def run_episode(episode_id, initial_agent, environment):
    """Run through a single episode to termination.  Returns the results for that episode."""
    logging.info('Starting episode {}'.format(episode_id))
    if episode_id % 5 == 0:  # Perform value iteration
        # TODO: Don't mutate agent and don't create an Agent object here
        uoms = [option.uom for option in initial_agent.options]
        computed_policy_theta = islice(iterate(lambda computed_policy_theta: initial_agent.descriptor.compute_policy(computed_policy_theta, initial_agent.descriptor.learning_rate, environment.reward_vector, uoms, environment.exhaustive_states), initial_agent.computed_policy), 5)
        initial_agent = Agent(initial_agent.descriptor, initial_agent.options, last(computed_policy_theta))
    for step_data in generate_state(initial_agent, environment):
        if step_data.state.is_terminal:
            agent = step_data.agent
            terminal_agent = agent.descriptor.terminal_update(agent, step_data.action, step_data.state)
            results = m(episode_id=episode_id, steps=step_data.step_id)
            assert step_data.state.position == Position(environment.size - 1, environment.size - 1)
            assert step_data.state.value == step_data.state.position.x + step_data.state.position.y * environment.size
            return EpisodeData(terminal_agent, results, episode_id)
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
