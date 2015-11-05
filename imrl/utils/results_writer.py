"""Write experimental results data to output files."""

# System
import logging
import os
from collections import namedtuple
from functools import reduce

# Third party
from pyrsistent import v, pmap
from cytoolz.itertoolz import interleave
from more_itertools import chunked
import numpy


ResultsDescriptor = namedtuple('ResultsDescriptor', ('interval', 'output_path', 'keys'))


def read_results(results_path):
    '''Read into a numpy array all the values in the given file path.'''
    return numpy.loadtxt(results_path, skiprows=1)


def results_logger():
    '''Construct and return the reuslts logger.'''
    logger = logging.getLogger("results_logger")
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def output_stdout(output_string, output_path):
    '''Given some output_string, print it to stdout and write it to the given output_path.'''
    results_logger().info(output_string)


def output_file(output_string, output_path):
    '''Given some output_string, write it to the given output_path.'''
    with open(output_path, 'a') as output_file:
        output_file.write(output_string + '\n')


def merge_results(results):
    '''
    Given a list of dictionary results from episodes and the interesting keys, merge them into a single dictionary.
    Example: [{episode_id: 1, steps: 22}, {episode_id: 2, steps: 30}] -> {episode_id: [1, 2], steps: [22, 30]}
    '''
    seed_dictionary = pmap({key: v() for key, _ in results[0].items()})
    return pmap(reduce(lambda result1, y: {key: value.append(y[key]) for key, value in result1.items()}, [seed_dictionary] + results))


def initialize_results(results_descriptor):
    '''Remove any conflicting results file and create the new one with its header.'''
    output_path = results_descriptor.output_path
    keys = results_descriptor.keys
    try:
        os.remove(output_path)
    except OSError:
        pass
    keys_string = ' '.join(key for key in keys)
    output_file(keys_string, output_path)


def write_results(results, results_descriptor):
    '''Output the given results to terminal and to file.'''
    output_path = results_descriptor.output_path
    keys = results_descriptor.keys
    value_vectors = (results[key] for key in keys)
    rows = chunked(interleave(value_vectors), len(keys))
    string_rows = map(lambda v: ' '.join(str(x) for x in v), rows)
    all_string_rows = '\n'.join(string_row for string_row in string_rows)
    keys_string = ' '.join(key for key in keys)
    output_stdout(keys_string + '\n' + all_string_rows, output_path)
    output_file(all_string_rows, output_path)
