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
    parser.add_argument('--num_steps', help='Number of steps for which to run the experiment.', type=int, default=100000)
    parser.add_argument('--log_level', help='Set log level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--results_interval', help='Log results out to terminal and file every n intervals.', type=int, default=5)
    parser.add_argument('--results_path', help='File path to save the results to.  Default is results.txt in the current working directory.', default=os.path.join(os.getcwd(), 'results.txt'))
    # Agent
    parser.add_argument('--alpha', help='Value iteration step size.', type=float, default=0.03)
    parser.add_argument('--eta', help='Option model step size.', type=float, default=0.03)
    parser.add_argument('--gamma', help='Discount factor.', type=float, default=0.99)
    parser.add_argument('--epsilon', help='New state sample distance threshold', type=float, default=0.1)
    parser.add_argument('--zeta', help='Intrinsic reward decay parameter.', type=float, default=0.01)
    parser.add_argument('--beta', help='RBF kernel width parameter.', type=float, default=80)
    parser.add_argument('--plan_interval', help='Execute value iteration after every n steps', type=int, default=1000)
    parser.add_argument('--num_vi', help='Number of iterations of value iteration to perform.', type=int, default=10)
    parser.add_argument('--agent_policy', help='Choose the agent\'s policy.', choices=['random'], default='random')
    parser.add_argument("--agent_viz", action='store_true', default=True, help="Plot agent statistics during runs?")
    # Environment
    parser.add_argument('--environment', help='Choose the environment.', choices=['gridworld', 'gridworld_continuous'], default='gridworld')
    parser.add_argument('--gridworld_size', help='Gridworld is size * size', type=int, default=10)
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
                  (args.environment == 'gridworld_continuous' and GridworldContinuous(0.2, 0.01))
    policy = (args.agent_policy == 'random' and RandomPolicy(environment.num_actions))
    fa = (args.environment == 'gridworld' and TabularFA(environment.size * environment.size)) or \
        (args.environment == 'gridworld_continuous' and RBF(2, 7, beta=args.beta))
    agent = Agent(policy, fa, environment.num_actions, args.alpha, args.gamma, args.eta,
                  args.zeta, args.epsilon, args.num_vi, subgoals=environment.create_subgoals())
    if args.agent_viz:
        agent.create_visualization(args.environment == 'gridworld' or args.environment == 'combo_lock', environment)
    results_descriptor = ResultsDescriptor(args.results_interval, args.results_path, ['interval_id', 'steps'])
    experiment_descriptor = ExperimentDescriptor(args.plan_interval, args.num_steps)
    start(experiment_descriptor, agent, environment, results_descriptor)


if __name__ == '__main__':
    main(sys.argv[1:])
