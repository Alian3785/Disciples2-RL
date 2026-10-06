import pytest

from campaign_fixtures import CampaignSnapshots


@pytest.fixture(scope="session")
def _campaign_snapshots():
    return CampaignSnapshots()


@pytest.fixture
def campaign_factory(_campaign_snapshots):
    """Fresh mutable state per call; expensive seeded initialization is shared."""
    opened = []

    def make(**kwargs):
        env = _campaign_snapshots.create(**kwargs)
        opened.append(env)
        return env

    yield make
    for env in opened:
        env.close()
