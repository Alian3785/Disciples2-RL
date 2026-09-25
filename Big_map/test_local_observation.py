"""Locality, radius, boundary, battle and paired transition regression checks."""
import copy
import os
import random

import numpy as np
from battle_env import BattleEnv
from train_campaign import make_env
from sb3_contrib.common.maskable.utils import get_action_masks

MAPS = [('orc_duel', None), ('small', 'dragon'), ('small', 'blue_dragon'),
        ('super_last_stand', None), ('magic_train', None)]


def make(name, objective, version):
    os.environ['CAMPAIGN_OBSERVATION_VERSION'] = version
    return make_env(map_name=name, campaign_objective=objective, use_boss_starting_roster=False)


def test_transitions():
    original = BattleEnv.__init__
    def seeded(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.seed(123)
    BattleEnv.__init__ = seeded
    try:
        for name, objective in MAPS:
            a, b = make(name, objective, 'baseline'), make(name, objective, 'local5')
            oa, _ = a.reset(seed=42)
            ob, _ = b.reset(seed=42)
            assert not a.unwrapped.use_boss_starting_roster
            assert not b.unwrapped.use_boss_starting_roster
            rng = np.random.default_rng(42)
            battles = 0
            for step in range(600):
                assert b.observation_space.contains(ob)
                mask = get_action_masks(a)
                np.testing.assert_array_equal(mask, get_action_masks(b))
                action = int(rng.choice(np.flatnonzero(mask)))
                py_rng, np_rng = random.getstate(), np.random.get_state()
                oa, ra, ta, tra, ia = a.step(action)
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                previous, reference = ob.copy(), ob
                ob, rb, tb, trb, ib = b.step(action)
                np.testing.assert_array_equal(previous, reference)
                assert (ra, ta, tra) == (rb, tb, trb), (name, step, ra, rb)
                assert a.unwrapped.grid_env.agent_pos == b.unwrapped.grid_env.agent_pos
                assert a.unwrapped.gold == b.unwrapped.gold
                ab = a.unwrapped.full_obs_slices['battle']
                bb = b.unwrapped._local_observation.slices['battle']
                np.testing.assert_array_equal(oa[slice(*ab)], ob[slice(*bb)])
                battles += int(b.unwrapped.mode == b.unwrapped.MODE_BATTLE)
                if ta or tra:
                    oa, _ = a.reset(seed=step)
                    ob, _ = b.reset(seed=step)
            print('TRANSITIONS', name, objective, a.observation_space.shape,
                  b.observation_space.shape, 'battle steps', battles, 'PASS', flush=True)
            a.close()
            b.close()
    finally:
        BattleEnv.__init__ = original


def test_locality():
    for name, objective in MAPS:
        env = make(name, objective, 'local5')
        env.reset(seed=42)
        e = env.unwrapped
        encoder, grid = e._local_observation, e.grid_env
        ax, ay = grid.agent_pos
        far = lambda p: max(abs(p[0] - ax), abs(p[1] - ay)) > 2
        original = encoder.build()
        # Mutate hidden world state, including already-visited cells, at once.
        grid.visited_cells.update((x, y) for x in range(grid.grid_size) for y in range(grid.grid_size))
        for enemy_id, pos in list(grid.enemy_positions.items()):
            if far(pos):
                grid.enemies_alive[enemy_id] = not grid.enemies_alive.get(enemy_id, False)
                grid.enemy_positions[enemy_id] = (grid.grid_size + 10, grid.grid_size + 10)
                e.enemy_team_states[enemy_id] = []
                e.enemy_map_spell_effects[enemy_id] = {'armor', 'damage'}
        for attr in ('chests', 'mana_sources'):
            mapping = getattr(e, attr)
            for pos in list(mapping):
                if far(pos):
                    del mapping[pos]
        for attr in ('obstacle_positions', 'dynamic_blocked_positions'):
            getattr(grid, attr).update((x, y) for x in range(grid.grid_size)
                                     for y in range(grid.grid_size) if far((x, y)))
        e.castle_heal_tiles = tuple(p for p in e.castle_heal_tiles if not far(p))
        e.grid_ruin_positions = tuple(p for p in e.grid_ruin_positions if not far(p))
        e.gold_mine_tiles = tuple(p for p in e.gold_mine_tiles if not far(p))
        e.legions_territory_tile_set = {p for p in e.legions_territory_tile_set if not far(p)}
        e.empire_territory_tile_set = {p for p in e.empire_territory_tile_set if not far(p)}
        e.trainer_interaction_tiles = tuple(p for p in e.trainer_interaction_tiles if not far(p))
        for prefix, stock in [('merchant', 'merchant_stocks'), ('spell_shop', 'spell_shop_stocks'),
                              ('mercenary', 'mercenary_site_rosters')]:
            sites = getattr(e, prefix + '_site_interaction_tiles')
            for site, tiles in list(sites.items()):
                if all(far(p) for p in tiles):
                    sites[site] = ()
                    getattr(e, stock).pop(site, None)
        np.testing.assert_array_equal(original, encoder.build())
        # The second ring must be represented; third ring must be invisible.
        env.reset(seed=42)
        e, grid = env.unwrapped, env.unwrapped.grid_env
        encoder = e._local_observation
        grid.agent_pos = (3, 3)
        grid.enemies_alive = {key: False for key in grid.enemies_alive}
        enemy_id = next(iter(grid.enemy_positions))
        grid.enemies_alive[enemy_id] = True
        grid.enemy_positions[enemy_id] = (5, 5)
        ring2 = encoder.build()
        tiles = ring2[slice(*encoder.slices['local_tiles'])].reshape(25, -1)
        assert tiles[24, encoder.TILE_FEATURES.index('enemy')] == 1
        grid.enemy_positions[enemy_id] = (6, 6)
        ring3 = encoder.build()
        grid.enemies_alive[enemy_id] = False
        np.testing.assert_array_equal(ring3, encoder.build())
        assert not np.array_equal(ring2, ring3)
        grid.agent_pos = (0, 0)
        tiles = encoder.tiles().reshape(25, -1)
        assert not tiles[:10].any() and not tiles[10:12].any()
        assert tiles[12, 0] == 1
        assert env.observation_space.contains(encoder.build())
        print('LOCALITY/RADIUS/BOUNDARY', name, objective, 'PASS', flush=True)
        env.close()
    os.environ.pop('CAMPAIGN_OBSERVATION_VERSION', None)
    env = make_env(map_name='orc_duel', use_boss_starting_roster=False)
    assert env.unwrapped.observation_version == 'local5'
    env.close()


if __name__ == '__main__':
    test_locality()
    test_transitions()
    print('ALL CHECKS PASSED', flush=True)
