"""Test the results writer."""

# System
import os

# Third Party
from pyrsistent import m, v
import numpy

# IMRL
from imrl.utils.results_writer import write_results, read_results, ResultsDescriptor, initialize_results


def test_results_writer():
    '''Does the results writer output what it should?'''
    results = m(episode=v(0, 1, 2), step_count=v(12, 22, 11))
    output_path = os.path.join(os.getcwd(), 'results.txt')
    results_descriptor = ResultsDescriptor(2, output_path, ['episode', 'step_count'])
    initialize_results(results_descriptor)
    write_results(results, results_descriptor)
    results_check = read_results(output_path)
    assert numpy.array_equal(results_check, numpy.array([[0., 12.], [1., 22.], [2., 11.]]))
    new_results = m(episode=v(3), step_count=v(12.33))
    write_results(new_results, results_descriptor)
    new_results_check = read_results(output_path)
    assert numpy.array_equal(new_results_check, numpy.array([[0., 12.], [1., 22.], [2., 11.], [3., 12.33]]))
