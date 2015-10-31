#!/usr/bin/env python3
"""Command line interface to execute experiments."""

# System
import sys
import argparse
import logging

def parse_args(argv):
    """Create command line arguments parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Seed with which to initialize random number generator.", type=float, nargs="+")
    parser.add_argument("--episodes", help="Number of episodes to run the experiment.", type=int, nargs="+")
    parser.add_argument("--log", help="Set log level.", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def log_level(level_string):
    """Take the log level string and return the corresponding log level value."""
    if level_string == "DEBUG":
        return logging.DEBUG
    elif level_string == "INFO":
        return logging.INFO
    elif level_string == "WARNING":
        return logging.WARNING


def main(argv):
    """Execute experiment."""
    args = parse_args(argv)
    logging.basicConfig(level=log_level(args.log))
    logging.info("Started execution.")


if __name__ == "__main__":
    main(sys.argv[1:])
