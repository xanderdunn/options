"""Module information"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


config = {
    'description': 'Intrinsically motivated, model-based hierarchical reinforcement learning agents.',
    'author': 'Chris Vigorito and Xander Dunn',
    'url': 'https://bitbucket.org/ChrisVigorito/imrl',
    'download_url': 'https://bitbucket.org/ChrisVigorito/imrl',
    'author_email': 'cmvigorito@gmail.com',
    'version': '0',
    'install_requires': ['pytest', 'numpy', 'scipy', 'scikit-learn', 'pyrsistent', 'cytoolz', 'more_itertools', 'matplotlib'],
    'packages': ['imrl'],
    'scripts': [],
    'name': 'Intrinsically Motivated Reinforcement Learning'
}

setup(**config)
