"""Test the agent-environment interface."""

from imrl.interface.experiment import episode_ids


def test_episode_generation():
    """Are episodes generated correctly?"""
    assert list(episode_ids(0)) == []
    assert list(episode_ids(1)) == [0]
    assert list(episode_ids(4)) == [0, 1, 2, 3]
