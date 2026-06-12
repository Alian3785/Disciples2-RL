import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


VALUABLE_ITEM_GOLD_VALUES = {
    "Bronze Ring (Valuable)": 250,
    "Silver Ring (Valuable)": 500,
    "Emerald (Valuable)": 750,
    "Gold Ring (Valuable)": 1000,
    "Ruby (Valuable)": 1250,
    "Sapphire (Valuable)": 1500,
    "Diamond (Valuable)": 1750,
    "Ancient Relic (Valuable)": 2000,
    "Royal Scepter (Valuable)": 2500,
    "Imperial Crown (Valuable)": 5000,
}


def _collect_adjacent_custom_chest(env: CampaignEnv, loot: tuple[str, ...]):
    start_x, start_y = env.grid_env.agent_pos
    chest_tile = (start_x + 2, start_y)
    move_target = (start_x + 1, start_y)
    env.grid_env.obstacle_positions.discard(move_target)
    env.grid_env.obstacle_positions.discard(chest_tile)
    env.chests = {chest_tile: loot}
    env.heroitems = []
    env.extra_healing_bottles = 0
    env.extra_revive_bottles = 0
    env._sync_grid_chest_positions()
    env.step(env.grid_env.ACTION_RIGHT)
    return move_target


def test_chest_loot_inventory_entries_keep_gold_values():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    move_target = _collect_adjacent_custom_chest(
        env,
        (
            env.INVULNERABILITY_POTION_ITEM_NAME,
            "Diamond (Valuable)",
            "Test Relic",
        ),
    )

    assert env.grid_env.agent_pos == move_target
    assert [str(entry) for entry in env.heroitems] == [
        env.INVULNERABILITY_POTION_ITEM_NAME,
        "Diamond (Valuable)",
        "Test Relic",
    ]
    assert [getattr(entry, "gold", None) for entry in env.heroitems] == [700, 1750, 0]


def test_all_game_valuables_keep_gold_values_when_collected():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    move_target = _collect_adjacent_custom_chest(env, tuple(VALUABLE_ITEM_GOLD_VALUES))

    assert env.grid_env.agent_pos == move_target
    assert [str(entry) for entry in env.heroitems] == list(VALUABLE_ITEM_GOLD_VALUES)
    assert [getattr(entry, "gold", None) for entry in env.heroitems] == list(
        VALUABLE_ITEM_GOLD_VALUES.values()
    )


def test_auto_consumed_chest_loot_granted_items_keep_gold_values():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    move_target = _collect_adjacent_custom_chest(
        env,
        (
            "Potion of Healing",
            "Life Potion",
        ),
    )

    assert env.grid_env.agent_pos == move_target
    assert [str(entry) for entry in env.heroitems] == [
        env.BONUS_SMALL_HEAL_ITEM_NAME,
        env.BONUS_REVIVE_ITEM_NAME,
    ]
    assert [getattr(entry, "gold", None) for entry in env.heroitems] == [150, 400]
