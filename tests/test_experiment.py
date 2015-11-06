"""Test the agent-environment interface."""

from imrl.interface.experiment import episode_id_generator


def test_episode_generation():
    """Are episodes generated correctly?"""
    assert list(episode_id_generator(0)) == []
    assert list(episode_id_generator(1)) == [0]
    assert list(episode_id_generator(4)) == [0, 1, 2, 3]
