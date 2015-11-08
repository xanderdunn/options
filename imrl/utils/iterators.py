"""Convenience functions on iterators."""


def iterate_results(func, x):
    """
    This is a copy of pytools.toolz.iterate except the initial value x is not returned, only func(x).
    Repeatedly apply a function func onto an original input

    Yields func(x), then func(func(x)), then func(func(func(x))), etc..
    """
    while True:
        x = func(x)
        yield x
