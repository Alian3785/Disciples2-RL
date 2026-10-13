"""The 24x24 small map keeps default's objects usable after shrinking."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grid import _expand_static_blocks, scale_static_tiles
from maps import get_map
from maps import small as small_map

SITE_KINDS = ("merchant", "spell_shop", "mercenary", "trainer")
TRAINER = "Тренировочный лагерь"


def site_buildings(env):
    for kind in SITE_KINDS:
        for name, data in getattr(env, f"BASE_{kind.upper()}_SITE_DATA").items():
            x, y = data["anchor"]
            building = set(scale_static_tiles(env.grid_size, tuple(
                (x + dx, y + dy) for dx in range(3) for dy in range(3))))
            yield kind, name, building, getattr(env, f"{kind}_site_interaction_tiles")[name]


def test_tile_helpers_match_environment_scaling():
    base_tiles = tuple((x, y) for x in range(48) for y in range(48))
    assert tuple(map(small_map._play_tile, base_tiles)) == scale_static_tiles(24, base_tiles)
    assert all(small_map._play_tile(small_map._base_tile((x, y))) == (x, y)
               for x in range(24) for y in range(24))


def test_every_site_has_a_solid_building_and_entrances_outside_it(campaign_factory):
    env = campaign_factory(map_name="small")
    obstacles = set(env._static_obstacle_tiles)
    for _, name, building, entrances in site_buildings(env):
        assert building <= obstacles, name
        assert entrances, name
        assert not building & set(entrances), name
    assert env.trainer_site_interaction_tiles[TRAINER] == ((21, 16), (19, 17), (20, 17))


def test_trainer_is_usable_from_its_entrance(campaign_factory):
    env = campaign_factory(map_name="small")
    env.grid_env.agent_pos = (21, 16)
    env.gold = 5000.0
    assert env._trainer_sites_at_position() == [TRAINER]
    mask = env.compute_action_mask()
    start = env.GRID_TRAINER_ACTION_START
    trainable = [i for i in range(len(env.TRAINER_POSITIONS)) if mask[start + i]]
    assert trainable
    env.step(start + trainable[0])
    assert env.gold < 5000.0


def test_objects_keep_their_own_tiles(campaign_factory):
    env = campaign_factory(map_name="small")
    labels = {}

    def put(tile, label):
        labels.setdefault(tuple(tile), []).append(label)

    for enemy_id, tile in env._static_enemy_positions.items():
        put(tile, f"enemy {enemy_id}")
    for tile in env._static_chests:
        put(tile, "chest")
    for tile in env._static_mana_sources:
        put(tile, "mana")
    for tile in env.gold_mine_tiles:
        put(tile, "mine")
    for kind, name, _, entrances in site_buildings(env):
        for tile in entrances:
            put(tile, kind)
    shared = {tile: names for tile, names in labels.items() if len(names) > 1}
    # City garrisons share the city tile; the spell-shop guard stands on its
    # corner entrance exactly as enemy 43 does on default.
    assert shared == {
        (14, 16): ["enemy 34", "enemy 68"],
        (20, 12): ["enemy 35", "enemy 69"],
        (23, 3): ["enemy 43", "spell_shop"],
    }
    assert env._static_chests[(13, 23)] == ("Banner of War",)


def test_field_stacks_stand_outside_mountains_and_buildings(campaign_factory):
    env = campaign_factory(map_name="small")
    solid = set(scale_static_tiles(24, _expand_static_blocks(env._map.obstacle_blocks)))
    kinds = env._map.enemy_location_kinds()
    inside = {enemy_id: tile for enemy_id, tile in env._static_enemy_positions.items()
              if kinds.get(enemy_id, "field") == "field" and tile in solid}
    assert inside == {}


def test_buildings_follow_the_objects_small_keeps():
    blocks = set(get_map("small").obstacle_blocks)
    # Spell shop and the kept ruin 70 keep their buildings.
    assert {(42, 2, 3, 3), (15, 9, 3, 3)} <= blocks
    # Removed villages and ruin 71 leave no empty buildings behind.
    assert not {(3, 1, 4, 4), (37, 6, 4, 4), (3, 20, 3, 3)} & blocks


def test_gold_mines_can_be_claimed(campaign_factory):
    env = campaign_factory(map_name="small")
    mines = set(env.gold_mine_tiles)
    assert mines == {(1, 15), (12, 2), (14, 23), (20, 23)}
    assert not mines & set(env.legions_territory_forbidden_tiles)
    assert not mines & set(env._build_empire_territory_forbidden_tiles())


@pytest.mark.parametrize("layer", ["water", "forest", "territory_forbidden"])
def test_terrain_layers_keep_default_proportions(layer):
    default, small = get_map("default"), get_map("small")
    default_share = len(getattr(default, f"{layer}_tiles_provider")()) / 48 ** 2
    small_tiles = set(scale_static_tiles(24, tuple(getattr(small, f"{layer}_tiles_provider")())))
    assert len(small_tiles) / 24 ** 2 <= 1.25 * default_share


@pytest.mark.parametrize("map_name,agent_mine,bot_mine",
                         [("default", (2, 31), (24, 4)), ("small", (1, 15), (12, 2))])
def test_capital_territories_stay_apart(campaign_factory, map_name, agent_mine, bot_mine):
    # Without ruin 70's building on small the Empire's land crossed into the
    # player's half and took all of it, home mine included, by day 15.
    env = campaign_factory(map_name=map_name)
    agent, bot = set(env.legions_territory_order), set(env.empire_territory_order)
    assert not agent & bot
    assert agent_mine in agent and bot_mine in bot
