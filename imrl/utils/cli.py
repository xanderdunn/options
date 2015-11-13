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
from imrl.agent.agent import Agent
from imrl.agent.policy.policy_random import RandomPolicy
from imrl.agent.fa.tabular import TabularFA
from imrl.agent.fa.rbf import RBF
from imrl.environment.gridworld import Gridworld
from imrl.environment.gridworld_continuous import GridworldContinuous
from imrl.utils.results_writer import ResultsDescriptor


def parse_args(argv):
    """Create command line arguments parser."""
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', help='Seed with which to initialize random number generator.', type=int)
    parser.add_argument('--episodes', help='Number of episodes to run the experiment.', type=int, default=200)
    parser.add_argument('--log_level', help='Set log level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--results_interval', help='Log results out to terminal and file every n episodes.', type=int, default=100)
    parser.add_argument('--results_path', help='File path to save the results to.  Default is results.txt in the current working directory.', default=os.path.join(os.getcwd(), 'results.txt'))
    # Agent
    parser.add_argument('--alpha', help='Value iteration step size.', type=float, default=0.1)
    parser.add_argument('--eta', help='Option model step size.', type=float, default=0.1)
    parser.add_argument('--gamma', help='Discount factor.', type=float, default=0.99)
    parser.add_argument('--epsilon', help='New state sample distance threshold', type=float, default=0.1)
    parser.add_argument('--vi_interval', help='Execute value iteration after every n episodes', type=int, default=50)
    parser.add_argument('--num_vi', help='Number of iterations of value iteration to perform at each interval vi_interval.', type=int, default=1)
    parser.add_argument('--agent_policy', help='Choose the agent\'s policy.', choices=['random'], default='random')
    parser.add_argument('--func_approx', help='Choose the agent\'s function approximator.', choices=['tabular', 'rbf'], default='tabular')
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
    environment = (args.environment == 'gridworld' and Gridworld(args.gridworld_size, args.failure_rate)) or \
                  (args.environment == 'gridworld_continuous' and GridworldContinuous(0.2, 0.05))
    policy = (args.agent_policy == 'random' and RandomPolicy(environment.num_actions))
    fa = (args.func_approx == 'tabular' and TabularFA(environment.size*environment.size)) or \
        (args.func_approx == 'rbf' and RBF(2, 5))
    samples = []  # list(range(environment.num_states()))
    samples.reverse()
    agent = Agent(policy, fa, environment.num_actions, args.alpha, args.gamma, args.eta, args.epsilon, samples)
    results_descriptor = ResultsDescriptor(args.results_interval, args.results_path, ['episode_id', 'steps'])
    experiment_descriptor = ExperimentDescriptor(args.num_vi, args.vi_interval, args.episodes)
    start(experiment_descriptor, agent, environment, results_descriptor)


if __name__ == '__main__':
    main(sys.argv[1:])
