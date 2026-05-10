import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _merchant_buy_action(env: CampaignEnv, item_index: int) -> int:
    return env.GRID_MERCHANT_BUY_ACTION_START + int(item_index)


def _potion_use_action(env: CampaignEnv, item_name: str, target_pos: int) -> int:
    return (
        env._potion_action_start_for_item(item_name)
        + env.GRID_POTION_USE_POSITIONS.index(target_pos)
    )


def _merchant_site_name(env: CampaignEnv, index: int) -> str:
    return list(env.merchant_site_interaction_tiles.keys())[int(index)]


def _merchant_tile(env: CampaignEnv, site_index: int) -> tuple[int, int]:
    site_name = _merchant_site_name(env, site_index)
    return tuple(env.merchant_site_interaction_tiles[site_name][0])


def test_merchant_auto_sale_sells_only_junk_items():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    merchant_tile = _merchant_tile(env, 0)
    env.grid_env.agent_pos = merchant_tile
    env.moves = 0
    env.gold = 100.0
    env.heroitems = [
        env._make_hero_item_entry("Ruby (Valuable)"),
        env._make_hero_item_entry(env.STRENGTH_POTION_ITEM_NAME),
        "Bronze Ring (Valuable)",
    ]

    _, reward, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    assert not terminated
    assert not truncated
    assert env.grid_env.agent_pos == merchant_tile
    assert info["is_at_merchant"] is True
    assert info["merchant_auto_sale"] is True
    assert info["merchant_sold_items_count"] == 2
    assert info["merchant_sale_gold"] == 1500.0
    assert reward == 2.0 * env.reward_sell_junk_item
    assert env.gold == 1600.0
    remaining_item_names = [str(entry) for entry in env.heroitems]
    assert env.STRENGTH_POTION_ITEM_NAME in remaining_item_names
    assert "Ruby (Valuable)" not in remaining_item_names
    assert "Bronze Ring (Valuable)" not in remaining_item_names
    assert [item["name"] for item in info["merchant_sold_items"]] == [
        "Ruby (Valuable)",
        "Bronze Ring (Valuable)",
    ]


def test_merchant_buy_actions_require_tile_gold_and_stock():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    life_action = _merchant_buy_action(env, 0)
    alara_tile = _merchant_tile(env, 0)
    tralara_tile = _merchant_tile(env, 1)
    alara_name = _merchant_site_name(env, 0)
    life_name = str(env.MERCHANT_BUY_ITEMS[0]["name"])

    mask = env.compute_action_mask()
    assert bool(mask[life_action]) is False

    env.grid_env.agent_pos = alara_tile
    env.gold = 400.0
    mask = env.compute_action_mask()
    assert bool(mask[life_action]) is True

    env.gold = 399.0
    mask = env.compute_action_mask()
    assert bool(mask[life_action]) is False

    env.gold = 400.0
    env.merchant_stocks[alara_name][life_name] = 0
    mask = env.compute_action_mask()
    assert bool(mask[life_action]) is False

    env.grid_env.agent_pos = tralara_tile
    env.gold = 400.0
    mask = env.compute_action_mask()
    assert bool(mask[life_action]) is False


def test_merchant_buys_feed_existing_heal_and_revive_actions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    alara_tile = _merchant_tile(env, 0)
    alara_name = _merchant_site_name(env, 0)
    env.grid_env.agent_pos = alara_tile
    env.gold = 2600.0

    small_action = _merchant_buy_action(env, 1)
    large_action = _merchant_buy_action(env, 2)
    ointment_action = _merchant_buy_action(env, 3)
    revive_action = _merchant_buy_action(env, 0)
    energy_action = _merchant_buy_action(env, 4)

    _, _, _, _, small_info = env.step(small_action)
    _, _, _, _, large_info = env.step(large_action)
    _, _, _, _, ointment_info_a = env.step(ointment_action)
    _, _, _, _, ointment_info_b = env.step(ointment_action)
    _, _, _, _, revive_info = env.step(revive_action)
    _, _, _, _, energy_info = env.step(energy_action)

    assert small_info["merchant_buy_purchased"] is True
    assert large_info["merchant_buy_purchased"] is True
    assert ointment_info_a["merchant_buy_purchased"] is True
    assert ointment_info_b["merchant_buy_purchased"] is True
    assert revive_info["merchant_buy_purchased"] is True
    assert energy_info["merchant_buy_purchased"] is True
    assert env.gold == 350.0
    assert env.merchant_stocks[alara_name][str(env.MERCHANT_BUY_ITEMS[1]["name"])] == 9
    assert env.merchant_stocks[alara_name][str(env.MERCHANT_BUY_ITEMS[2]["name"])] == 9
    assert env.merchant_stocks[alara_name][str(env.MERCHANT_BUY_ITEMS[3]["name"])] == 8
    assert env.merchant_stocks[alara_name][str(env.MERCHANT_BUY_ITEMS[0]["name"])] == 9
    assert env.merchant_stocks[alara_name][str(env.MERCHANT_BUY_ITEMS[4]["name"])] == 0
    assert env.extra_healing_bottles == 1
    assert env.extra_heal_bottles == 1
    assert env.extra_revive_bottles == 1
    assert env.get_bottle_inventory_counters()["healing_left"] == 4
    assert env.get_bottle_inventory_counters()["heal_left"] == 4
    assert env.get_bottle_inventory_counters()["revive_left"] == 4
    assert env.get_bottle_inventory_counters()["ointment_left"] == 2
    assert str(env.MERCHANT_BUY_ITEMS[4]["name"]) in [str(entry) for entry in env.heroitems]
    assert energy_info["merchant_auto_sale"] is False

    blue_state = env._get_blue_state()
    target = max(
        (
            unit
            for unit in blue_state
            if int(unit.get("position", -1)) in env.GRID_BOTTLE_POSITIONS
        ),
        key=lambda unit: float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0.0),
    )
    target_pos = int(target.get("position", -1))
    max_hp = float(target.get("maxhp", 0) or target.get("max_health", 0) or 0.0)
    max_hp = max(300.0, max_hp)
    target["maxhp"] = max_hp
    target["max_health"] = max_hp
    heal_ointment_action = _potion_use_action(env, env.HEALING_OINTMENT_ITEM_NAME, target_pos)
    heal_large_action = _potion_use_action(env, env.BONUS_LARGE_HEAL_ITEM_NAME, target_pos)
    heal_small_action = _potion_use_action(env, env.BONUS_SMALL_HEAL_ITEM_NAME, target_pos)

    target["hp"] = max_hp - 220.0
    target["health"] = target["hp"]
    _, _, _, _, heal_ointment_info = env.step(heal_ointment_action)
    assert heal_ointment_info["selected_heal_bottle_kind"] == env.HEALING_OINTMENT_ITEM_NAME
    assert env.get_bottle_inventory_counters()["ointment_left"] == 1

    target["hp"] = max_hp - 80.0
    target["health"] = max_hp - 80.0
    _, _, _, _, heal_large_info = env.step(heal_large_action)
    assert heal_large_info["selected_heal_bottle_kind"] == env.BONUS_LARGE_HEAL_ITEM_NAME
    assert env.get_bottle_inventory_counters()["heal_left"] == 3

    target["hp"] = max_hp - 40.0
    target["health"] = max_hp - 40.0
    _, _, _, _, heal_small_info = env.step(heal_small_action)
    assert heal_small_info["selected_heal_bottle_kind"] == env.BONUS_SMALL_HEAL_ITEM_NAME
    assert env.get_bottle_inventory_counters()["healing_left"] == 3

    env.heal_bottles_used = env._max_heal_bottles_available()
    env.healing_bottles_used = env._max_healing_bottles_available()
    target["hp"] = max_hp - 80.0
    target["health"] = max_hp - 80.0
    _, _, _, _, heal_fallback_ointment_info = env.step(heal_ointment_action)
    assert heal_fallback_ointment_info["selected_heal_bottle_kind"] == env.HEALING_OINTMENT_ITEM_NAME
    assert env.get_bottle_inventory_counters()["ointment_left"] == 0

    target["hp"] = 0.0
    target["health"] = 0.0
    revive_use_action = env.GRID_REVIVE_ACTION_START + env.REVIVE_BOTTLE_POSITIONS.index(target_pos)
    _, _, _, _, revive_use_info = env.step(revive_use_action)
    assert revive_use_info["revived"] is True
    assert env.get_bottle_inventory_counters()["revive_left"] == 3
