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
from imrl.environment.combination_lock import CombinationLock
from imrl.utils.results_writer import ResultsDescriptor
from imrl.interface.experiment2 import Experiment2


def parse_args(argv):
    """Create command line arguments parser."""
    alpha = 1
    eta = 1
    zeta = 1
    gamma = 0.99
    epsilon = 0.1
    beta = 80
    plan_interval = 10
    num_vi = 1
    retain_theta = True
    sim_samples = 25
    sim_steps = 25
    agent_viz = True
    viz_steps = 100
    environment = 'gridworld'
    gridworld_width = 5
    gridworld_height = 10
    failure_rate = 0

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', help='Seed with which to initialize random number generator.', type=int)
    parser.add_argument('--num_steps', help='Number of steps for which to run the experiment.', type=int, default=1000000)
    parser.add_argument('--log_level', help='Set log level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARNING')
    parser.add_argument('--results_interval', help='Log results out to terminal and file every n intervals.', type=int, default=50)
    parser.add_argument('--results_path', help='File path to save the results to. Default is results.txt in the current working directory.',
                        default=os.path.join(os.getcwd(), 'results.txt'))

    # Agent
    parser.add_argument('--alpha', help='Value iteration step size.', type=float, default=alpha)
    parser.add_argument('--eta', help='Option model step size.', type=float, default=eta)
    parser.add_argument('--gamma', help='Discount factor.', type=float, default=gamma)
    parser.add_argument('--epsilon', help='New state sample distance threshold', type=float, default=epsilon)
    parser.add_argument('--zeta', help='Intrinsic reward decay parameter.', type=float, default=zeta)
    parser.add_argument('--beta', help='RBF kernel width parameter.', type=float, default=beta)
    parser.add_argument('--plan_interval', help='Execute value iteration after every n steps', type=int, default=plan_interval)
    parser.add_argument('--num_vi', help='Number of iterations of value iteration to perform.', type=int, default=num_vi)
    parser.add_argument("--retain_theta", action='store_true', default=retain_theta, help="Retain previous value function estimate for VI.")
    parser.add_argument('--sim_samples', help='Number of sample start states from which to simulate options.', type=int, default=sim_samples)
    parser.add_argument('--sim_steps', help='Number of for which to simulate options.', type=int, default=sim_steps)
    parser.add_argument('--agent_policy', help='Choose the agent\'s initial policy.', choices=['random'], default='random')
    parser.add_argument("--agent_viz", action='store_true', default=agent_viz, help="Plot agent statistics during runs?")
    parser.add_argument('--viz_steps', help='Frequency with which to update agent visualization.', type=int, default=viz_steps)

    # Environment
    parser.add_argument('--environment', help='Choose the environment.',
                        choices=['gridworld', 'gridworld_continuous', 'combo_lock'], default=environment)
    parser.add_argument('--gridworld_width', help='Gridworld is width * height', type=int, default=gridworld_width)
    parser.add_argument('--gridworld_height', help='Gridworld is width * height', type=int, default=gridworld_height)
    parser.add_argument('--failure_rate', help='The percent of actions in this environment that fail.', type=float, default=failure_rate)
    return parser.parse_args(argv)


def log_level(level_string):
    """Take the log level string and return the corresponding log level value."""
    if level_string == 'DEBUG':
        return logging.DEBUG
    elif level_string == 'INFO':
        return logging.INFO
    elif level_string == 'WARNING':
        return logging.WARNING
    elif level_string == 'ERROR':
        return logging.ERROR


def main(argv):
    """Execute experiment."""
    args = parse_args(argv)
    random.seed(args.seed)
    logging.basicConfig(level=log_level(args.log_level))
    environment = (args.environment == 'gridworld' and Gridworld(args.gridworld_width, args.gridworld_height, args.failure_rate)) or \
                  (args.environment == 'gridworld_continuous' and GridworldContinuous(0.1, 0.01)) or \
                  (args.environment == 'combo_lock' and CombinationLock(args.gridworld_height, args.gridworld_width, 4, args.failure_rate))
    policy = (args.agent_policy == 'random' and RandomPolicy(environment.num_actions))
    fa = ((args.environment == 'gridworld' or args.environment == 'combo_lock') and
          TabularFA(environment.num_states(), environment.num_actions)) or \
        (args.environment == 'gridworld_continuous' and RBF(2, 7, environment.num_actions, beta=args.beta))
    agent = Agent(policy, fa, environment.num_actions, args.alpha, args.gamma, args.eta, args.zeta, args.epsilon,
                  args.num_vi, args.sim_samples, args.sim_steps, retain_theta=args.retain_theta, subgoals=environment.create_subgoals())
    if args.agent_viz:
        agent.create_visualization(args.environment == 'gridworld' or args.environment == 'combo_lock', environment)
    # agent.exploit(np.asarray([1, 1]))
    # agent.exploit(environment.num_states()-1)
    # results_descriptor = ResultsDescriptor(args.results_interval, args.results_path, ['interval_id', 'steps'])
    # experiment_descriptor = ExperimentDescriptor(args.plan_interval, args.num_steps)
    # start(experiment_descriptor, agent, environment, results_descriptor)
    e = Experiment2(agent, environment, args.plan_interval, args.num_steps, args.viz_steps)
    e.run()


if __name__ == '__main__':
    main(sys.argv[1:])
