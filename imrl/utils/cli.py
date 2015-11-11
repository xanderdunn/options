#!/usr/bin/env python3
"""Command line interface to execute experiments."""

# System
import sys
import argparse
import logging
import random
import os

# IMRL
from imrl.interface.experiment import start, ExperimentDescriptor
from imrl.environment.gridworld import gridworld_discrete
from imrl.environment.gridworld_continuous import gridworld_continuous
from imrl.agent.agent import agent_random_tabular
from imrl.utils.results_writer import ResultsDescriptor


def parse_args(argv):
    """Create command line arguments parser."""
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', help='Seed with which to initialize random number generator.', type=int)
    parser.add_argument('--episodes', help='Number of episodes to run the experiment.', type=int, default=1000)
    parser.add_argument('--log_level', help='Set log level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--results_interval', help='Log results out to terminal and file every n episodes.', type=int, default=100)
    parser.add_argument('--results_path', help='File path to save the results to.  Default is results.txt in the current working directory.', default=os.path.join(os.getcwd(), 'results.txt'))
    # Agent
    parser.add_argument('--alpha', help='Learning rate of the agent.', type=float, default=1.0)
    parser.add_argument('--eta', help='Learning rate of the options\' UOMs', type=float, default=1.0)
    parser.add_argument('--gamma', help='Discount factor of the option\'s UOMs', type=float, default=0.999)
    parser.add_argument('--vi_interval', help='Execute value iteration after every n episodes', type=int, default=5)
    parser.add_argument('--num_vi', help='Number of iterations of value iteration to perform at each interval vi_interval.', type=int, default=1)
    parser.add_argument('--vi_ex_start', help='Begin to execute the value iteration policy after n episodes.', type=int, default=None)
    parser.add_argument('--agent_policy', help='Choose the agent\'s policy.', choices=['random'], default='random')
    # Environment
    parser.add_argument('--environment', help='Choose the environment.', choices=['gridworld', 'gridworld_continuous'], default='gridworld')
    parser.add_argument('--gridworld_size', help='Gridworld is size * size', type=int, default=3)
    parser.add_argument('--failure_rate', help='The percent of actions in this environment that fail.', type=float, default=0.0)
    return parser.parse_args(argv)


def log_level(level_string):
    """Take the log level string and return the corresponding log level value."""
    if level_string == 'DEBUG':
        return logging.DEBUG
    elif level_string == 'INFO':
        return logging.INFO
    elif level_string == 'WARNING':
        return logging.WARNING


def main(argv):
    """Execute experiment."""
    args = parse_args(argv)
    random.seed(args.seed)
    logging.basicConfig(level=log_level(args.log_level))
    environment = (args.environment == 'gridworld' and gridworld_discrete(args.gridworld_size, args.failure_rate)) or \
                  (args.environment == 'gridworld_continuous' and gridworld_continuous(0.05, 0.01))
    agent = (args.agent_policy == 'random' and agent_random_tabular(args.gridworld_size ** 2, environment.num_actions, args.alpha, args.eta, args.gamma))
    results_descriptor = ResultsDescriptor(args.results_interval, args.results_path, ['episode_id', 'steps'])
    experiment_descriptor = ExperimentDescriptor(args.num_vi, args.vi_interval, args.episodes, args.vi_ex_start)
    start(experiment_descriptor, agent, environment, results_descriptor)


if __name__ == '__main__':
    main(sys.argv[1:])
