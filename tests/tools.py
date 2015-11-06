"""Test the combination lock."""


def generate_test(repeated_function, limit):
    """Generate an arbitrary function limit times."""
    for _ in range(limit):
        yield repeated_function()


def ratio_test(predicate, repeated_function, limit):
    """Perform the function up to the limit number of times and return the ratio of runs that satisfied the predicate"""
    return sum(map(predicate, generate_test(repeated_function, limit))) / limit
