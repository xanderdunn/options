# Intrinsically Motivated Reinforcement Learning

## Setup and Run
- Install dependencies: `pip install .` to install all packages defined in setup.py
- Run unit tests: `py.test -vv -s`
- Execute an experiment with `./imrl/utils/cli.py --episodes 6`

### Common Experiments
- Discrete, tabular, deterministic 3x3 gridworld: `./imrl/utils/cli.py --environment=gridworld --gridworld_size=3 --num_vi=1 --vi_ex_start=100 --episodes=200`
    - This learns the policy through value iteration for 100 episodes and then executes on that policy
- Discrete, tabular, stochastic 3x3 gridworld: `./imrl/utils/cli.py --environment=gridworld --gridworld_size=3 --num_vi=1 --vi_ex_start=100 --failure_rate=0.1 --episodes=200`
- Discrete, tabular, stochastic 5x5 gridworld testing theta convergence on a single value iteration: `./imrl/utils/cli.py --environment=gridworld --gridworld_size=5 --num_vi=1 --vi_interval=400 --failure_rate=0.1 --episodes=600 --vi_ex_start=401`

## Conventions
- Naming convention: Where variables are named that represent different time steps, such as a state `s` and next state `s'`, the first state will be named `state` and the second `state_prime`
