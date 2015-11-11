"""Managing data structures and algorithms used by an IMRL agent."""

# System
from collections import namedtuple

# Third party
from pyrsistent import pvector

# First party
from imrl.agent.policy_agent import policy_random
from imrl.agent.option import option_primitive
from imrl.agent.fa.func_approx import tabular_function_approximator
from imrl.agent.uom import update_m, update_u, UOM
from imrl.agent.option import Option
from imrl.agent.value_iteration import update_theta, initial_theta


# TODO: I need types that I can inherit from
AgentDescriptor = namedtuple('AgentDescriptor', ('policy', 'decide_action', 'num_states', 'feature_vector', 'update', 'compute_policy', 'learning_rate', 'terminal_update', 'switch_policy'))
Agent = namedtuple('Agent', ('descriptor',  # An AgentDescriptor object that describes all the agent's parameters
                             'options',     # A PVector of options, one for each action in the environment this agent operates in
                             'computed_policy'))     # A value iteration-computed policy


def switch_policy(agent, new_policy):
    """Given an agent and a new policy, switch the agent over to using the new policy."""
    descriptor = agent.descriptor
    new_descriptor = AgentDescriptor(new_policy, descriptor.decide_action, descriptor.num_states, descriptor.feature_vector, descriptor.update, descriptor.compute_policy, descriptor.learning_rate, descriptor.terminal_update, descriptor.switch_policy)
    return Agent(new_descriptor, agent.options, agent.computed_policy)


def agent_random_tabular(num_states, num_actions, learning_rate, eta, gamma):
    """An agent with a random policy."""
    descriptor = AgentDescriptor(policy_random, decide_action, num_states, tabular_function_approximator, update_options_agent, update_theta, learning_rate, terminal_update, switch_policy)
    options = pvector([option_primitive(num_states, eta, gamma) for i in range(num_actions)])
    return Agent(descriptor, options, initial_theta(num_states))


def terminal_update(agent, action, state):
    """Called to do any update of the termination state."""
    fv = agent.descriptor.feature_vector(state, agent.descriptor.num_states)
    options = agent.options
    option = options[action]
    uom = option.uom
    u_prime = update_u(uom, fv)
    return updated_agent(agent, action, option, uom, u_prime, uom.m)


def update_options_agent(agent, action, state, state_prime, tau):
    """Update an agent with options and return the new agent."""
    fv = agent.descriptor.feature_vector(state, agent.descriptor.num_states)
    fv_prime = agent.descriptor.feature_vector(state_prime, agent.descriptor.num_states)
    options = agent.options
    option = options[action]
    uom = option.uom
    m_prime = update_m(uom, fv, fv_prime, tau)
    u_prime = update_u(uom, fv)
    return updated_agent(agent, action, option, uom, u_prime, m_prime)


def updated_agent(agent, action, option, uom, u, m):
    """Convenience function to construct a new Agent given new u and m model matrices."""
    uom_prime = UOM(uom.descriptor, m, u)
    option_prime = Option(option.descriptor, uom_prime)
    options_prime = agent.options.set(action, option_prime)
    return Agent(agent.descriptor, options_prime, agent.computed_policy)


def decide_action(agent, state, num_actions, reward_function):
    """Receive the environment's latest state and return an action to take."""
    return agent.descriptor.policy(agent, state, num_actions, reward_function)
