"""Module information"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


config = {
    'description': 'Experiments with intrinsically motivated reinforcement learners.',
    'author': 'Chris Vigorito and Xander Dunn',
    'url': 'https://bitbucket.org/ChrisVigorito/imrl',
    'download_url': 'https://bitbucket.org/ChrisVigorito/imrl',
    'author_email': 'cmvigorito@gmail.com',
    'version': '0',
    'install_requires': ['pytest', 'numpy', 'scipy', 'scikit-learn'],
    'packages': ['imrl'],
    'scripts': [],
    'name': 'Intrinsicly Motivated Reinforcement Learning'
}

setup(**config)
