"""Independent copies of seeded campaigns for tests of post-reset behavior.

Only opted-in tests use snapshots. Constructor/reset tests still build real
environments. Snapshots exist in memory for one pytest worker, never on disk.
"""
import io
import pickle

from campaign_env import CampaignEnv
from grid_world_env import _TrackedEnemyPositionDict


def _reduce_enemy_positions(value):
    # Initialize the callback attribute before pickle inserts dictionary items;
    # restore the callback afterwards, bound to the NEW grid instance.
    return type(value), (), vars(value), None, iter(value.items())


class CampaignSnapshots:
    def __init__(self):
        self._snapshots = {}

    def create(self, *, cls=CampaignEnv, seed=42, **kwargs):
        options = dict(observation_version="local5", scripted_capital_bot_enabled=False,
                       use_boss_starting_roster=False, log_enabled=False)
        options.update(kwargs)
        key = (cls, seed, tuple(sorted(options.items())))
        if key not in self._snapshots:
            env = cls(**options)
            try:
                env.reset(seed=seed)
                stream = io.BytesIO()
                writer = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
                writer.dispatch_table = {_TrackedEnemyPositionDict: _reduce_enemy_positions}
                writer.dump(env)
                self._snapshots[key] = stream.getvalue()
            finally:
                env.close()
        return pickle.loads(self._snapshots[key])
