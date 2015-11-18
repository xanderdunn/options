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


ExperimentDescriptor = namedtuple('ExperimentDescriptor', ('plan_interval',      # Do value iteration updates every n steps
                                                           'num_steps'))         # Total number of steps to execute
IntervalData = namedtuple('IntervalData', ('agent', 'results', 'interval_id'))
StepData = namedtuple('StepData', ('state',    # The environment's current state
                                   'action',   # The action the agent took that got the environment into this state
                                   'step_id',  # The time step of this state
                                   'agent'))   # The current agent after having received reward and updated for this state


def run_step(step_data, environment):
    """Given the current state, choose an action to take and return the new environment state."""
    state = step_data.state
    agent = step_data.agent
    action = agent.policy.choose_action(state)
    state_prime = environment.next_state(state, action)
    agent.update(state, action, state_prime)
    return StepData(state_prime, action, step_data.step_id + 1, agent)


def generate_state(agent, environment):
    """Run a single planning interval for the given number of steps."""
    initial_step_data = StepData(environment.initial_state(), None, 0, agent)  # Initial state with step_id 0
    return iterate(lambda step_data: run_step(step_data, environment), initial_step_data)


def run_interval(interval_id, initial_agent, environment, interval_steps):
    """Run through a single planning interval.  Returns the results for that interval."""
    if interval_id > 0:
        initial_agent.policy = initial_agent.vi_policy
    logging.info('Starting interval {}'.format(interval_id))
    for step_data in generate_state(initial_agent, environment):
        if step_data.step_id == interval_steps:
            logging.info('Planning...')
            initial_agent.plan()
            if initial_agent.viz:
                initial_agent.viz.update()
            agent = step_data.agent
            results = m(interval_id=interval_id, steps=step_data.step_id)
            return IntervalData(agent, results, interval_id)


def interval_results_generator(agent, environment, experiment_description):
    """Execute planning intervals and yield the results for each interval."""
    return iterate_results(lambda interval_data: run_interval(interval_data.interval_id + 1, interval_data.agent, environment,
                                                             experiment_description.plan_interval), IntervalData(agent, None, 0))


# Think about using tee and chain, or izip to generate the step_id along with the interval_id
def start(experiment_description, agent, environment, results_descriptor):
    """Kick off the execution of an experiment."""
    initialize_results(results_descriptor)
    interval_results = islice(interval_results_generator(agent, environment, experiment_description), experiment_description.num_steps)
    results_interval_chunks = chunked(interval_results, results_descriptor.interval)
    for chunk in results_interval_chunks:
        results = [interval_data.results for interval_data in chunk]
        write_results(merge_results(results), results_descriptor)
