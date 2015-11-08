"""Managing data structures and algorithms used by an IMRL agent."""

# System
from collections import namedtuple

# Third party
from pyrsistent import pvector

# First party
from imrl.agent.policy import policy_random
from imrl.agent.option import option_primitive
from imrl.agent.fa.func_approx import tabular_function_approximator
from imrl.agent.uom import update_m, update_u, UOM
from imrl.agent.option import Option


# TODO: How will I handle the fact that only the special discrete tabular case will know the number states?
AgentDescriptor = namedtuple('AgentDescriptor', ('policy', 'decide_action', 'num_states', 'feature_vector', 'update'))
Agent = namedtuple('Agent', ('descriptor',  # An AgentDescriptor object that describes all the agent's parameters
                             'options'))    # A PVector of options, one for each action in the environment this agent operates in


def agent_random_tabular(num_states, num_actions):
    """An agent with a random policy."""
    descriptor = AgentDescriptor(policy_random, decide_action, num_states, tabular_feature_vector, update_options_agent)
    options = pvector([option_primitive(num_states) for i in range(num_actions)])
    return Agent(descriptor, options)


def tabular_feature_vector(state, num_states):
    """Given some state, return the tabular \"function approximation\" of it."""
    return tabular_function_approximator(state, num_states)


def update_options_agent(agent, action, state, state_prime):
    """Update an agent with options and return the new agent."""
    fv = agent.descriptor.feature_vector(state.value, agent.descriptor.num_states)
    fv_prime = agent.descriptor.feature_vector(state_prime.value, agent.descriptor.num_states)
    options = agent.options
    option = options[action]
    uom = option.uom
    m_prime = update_m(uom, fv, fv_prime)
    u_prime = update_u(uom, fv)
    assert not state.is_terminal
    uom_prime = UOM(uom.descriptor, m_prime, u_prime)
    option_prime = Option(option.descriptor, uom_prime)
    options_prime = options.set(action, option_prime)
    return Agent(agent.descriptor, options_prime)


def decide_action(policy, state, num_actions):
    """Receive the environment's latest state and return an action to take."""
    return policy(state, num_actions)
