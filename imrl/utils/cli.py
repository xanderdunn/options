#!/usr/bin/env python3
"""Command line interface to execute experiments."""

# System
import sys
import argparse
import logging
import random
import os

# IMRL
from imrl.interface import experiment
import imrl.environment.gridworld as gw
from imrl.agent.agent import RandomAgent, policy_random, decide_action
from imrl.utils.results_writer import ResultsDescriptor


def parse_args(argv):
    """Create command line arguments parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Seed with which to initialize random number generator.', type=int)
    parser.add_argument('--results_interval', help='Log results out to terminal and file every n episodes.', type=int, default=100)
    parser.add_argument('--results_path', help='File path to save the results to.  Default is results.txt in the current working directory.', default=os.path.join(os.getcwd(), 'results.txt'))
    parser.add_argument('--episodes', help="Number of episodes to run the experiment.", type=int, default=10000)
    parser.add_argument('--log_level', help='Set log level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--environment', help='Choose the environment.', choices=['gridworld'], default='gridworld')
    parser.add_argument('--agent_policy', help='Choose the agent\'s policy.', choices=['random'], default='random')
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
    agent = (args.agent_policy == 'random' and RandomAgent(policy_random, decide_action))
    environment = (args.environment == 'gridworld' and gw.Gridworld(10, 0.1, gw.take_action, 4, gw.initial_state))
    results_descriptor = ResultsDescriptor(args.results_interval, args.results_path, ['episode_id', 'steps'])
    experiment.start(args.episodes, agent, environment, results_descriptor)


if __name__ == '__main__':
    main(sys.argv[1:])
