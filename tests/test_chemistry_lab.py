"""Test the chemistry lab environment."""

# First party
from imrl.environment.chemistry_lab import ChemistryLab, SubgoalNode


def test_within_subgoals():
    """Test that it correctly identifies a state between or not between two subgoals."""
    a = SubgoalNode([0.5, 0.25])
    b = SubgoalNode([0.25, 0.25])
    c = SubgoalNode([0.5, 0.75])
    a.connections += [b, c]
    print(a.connections)

    assert not ChemistryLab.is_within_subgoals([0.5, 0.5], a, b)
    assert ChemistryLab.is_within_subgoals([0.3, 0.25], a, b)
    assert ChemistryLab.is_within_subgoals([0.2, 0.25], a, b)
    assert not ChemistryLab.is_within_subgoals([0.145, 0.25], a, b)
    assert ChemistryLab.is_within_subgoals([0.6, 0.6], a, c)


def test_subgoals():
    """Test that the created subgoals are correct."""
    root_subgoal = ChemistryLab.create_subgoals()
    assert len(root_subgoal.connections) == 2
    assert root_subgoal.root


def test_is_terminal():
    """Test that the state goes terminal when the agent makes a move outside the corridors."""
    lab = ChemistryLab()
    root_subgoal = ChemistryLab.create_subgoals()

    assert not lab.is_terminal(lab.initial_state(), root_subgoal)
    assert not lab.is_terminal([0.55, 0.4], root_subgoal)
    assert not lab.is_terminal([0.25, 0.4], root_subgoal)
    assert lab.is_terminal([0, 0], root_subgoal)
    assert lab.is_terminal([0.25, 0.64], root_subgoal)
    assert lab.is_terminal([1, 1], root_subgoal)
    assert not lab.is_terminal([0.64, 0.75], root_subgoal)
    assert not lab.is_terminal([0.64, 0.70], root_subgoal)
    assert not lab.is_terminal([0.25, 0.70], root_subgoal)
    assert lab.is_terminal([0.64, 0.5], root_subgoal)
