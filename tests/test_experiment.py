"""Test the combination lock."""

from imrl.interface.experiment import generate_episode_id


def test_episode_generation():
    """Are episodes generated correctly?"""
    assert list(generate_episode_id(0)) == []
    assert list(generate_episode_id(1)) == [0]
    assert list(generate_episode_id(4)) == [0, 1, 2, 3]
