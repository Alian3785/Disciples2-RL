import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.trade_train import (
    ELITE_MERCENARIES,
    ENEMY_STACKS,
    MERCENARY_CAMP_NAME,
    MERCHANT_NAME,
    POWERFUL_MERCHANT_ITEMS,
    STARTING_GOLD,
    TRAINER_NAME,
)


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="trade_train",
        log_enabled=False,
        persist_blue_hp=False,
        reward_needaunit_turn_penalty=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[dict]:
    return [
        unit
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def test_trade_train_is_registered_with_duke_gold_and_three_trade_sites():
    assert "trade_train" in available_maps()
    map_config = get_map("trade_train")

    assert map_config.grid_size == 48
    assert map_config.play_grid_size == 24
    assert map_config.default_objective == "all_enemies"
    assert map_config.starting_roster == {8: "Герцог"}
    assert map_config.starting_gold == STARTING_GOLD == 9000.0
    assert len(map_config.enemy_stacks) == 6
    assert tuple(map_config.merchant_buy_items or ()) == POWERFUL_MERCHANT_ITEMS
    assert set(map_config.merchant_sites) == {MERCHANT_NAME}
    assert set(map_config.mercenary_sites) == {MERCENARY_CAMP_NAME}
    assert set(map_config.trainer_sites) == {TRAINER_NAME}
    assert map_config.chests == ()
    assert map_config.mana_sources == ()
    assert map_config.spell_shop_sites == {}

    env = _make_env(Realcapital=1, typeoflord=3, use_boss_starting_roster=True)
    env.reset(seed=123)

    assert env.grid_size == 24
    assert env.Realcapital == 2
    assert env.typeoflord == 1
    assert env.grid_env.agent_pos == (2, 13)
    assert env.gold == pytest.approx(9000.0)
    assert [
        (int(unit["position"]), str(unit["name"]))
        for unit in _living_units(env.blue_team_state)
    ] == [(8, "Герцог")]

    site_anchor_xs = {
        int(env.merchant_site_anchors[MERCHANT_NAME][0]),
        int(env.mercenary_site_anchors[MERCENARY_CAMP_NAME][0]),
        int(env.trainer_site_anchors[TRAINER_NAME][0]),
    }
    assert min(site_anchor_xs) > int(env.grid_env.agent_pos[0])


def test_trade_train_merchant_sells_all_powerful_items_and_agent_can_buy_artifact():
    env = _make_env()
    env.reset(seed=123)

    expected_names = tuple(
        str(item_data["name"])
        for item_data in POWERFUL_MERCHANT_ITEMS
    )
    assert env.scenario_merchant_item_names == expected_names
    assert env.GRID_MERCHANT_BUY_ACTION_COUNT == len(expected_names)
    assert env.merchant_stocks[MERCHANT_NAME] == {
        str(item_data["name"]): int(item_data["stock"])
        for item_data in POWERFUL_MERCHANT_ITEMS
    }
    assert env.scenario_merchant_potion_item_names == (
        "Эликсир Всевышнего",
        "Эликсир неуязвимости",
        "Эликсир быстроты",
    )
    assert env.merchant_stocks[MERCHANT_NAME]["Эликсир неуязвимости"] == 2
    assert env.merchant_stocks[MERCHANT_NAME]["Эликсир быстроты"] == 2
    assert env.HASTE_ELIXIR_INITIATIVE_MULTIPLIER == pytest.approx(1.60)
    assert "Angel Orb" in env.scenario_battle_magic_item_names
    assert "Elder Vampire Talisman" in env.scenario_battle_magic_item_names
    assert "Elder Vampire Orb" in env.scenario_battle_magic_item_names
    assert "Venerable Warrior Orb" in env.scenario_battle_magic_item_names
    assert env.scenario_book_item_names == ("Tome of Sorcery",)

    env.grid_env.agent_pos = env.merchant_site_interaction_tiles[MERCHANT_NAME][0]
    buy_action = env.GRID_MERCHANT_BUY_ACTION_START
    assert bool(env.compute_action_mask()[buy_action]) is True

    _, _, terminated, truncated, info = env.step(buy_action)

    assert terminated is False
    assert truncated is False
    assert info["merchant_buy_purchased"] is True
    assert info["merchant_buy_item"] == "Bethrezen's Claw (Artifact)"
    assert info["merchant_buy_site"] == MERCHANT_NAME
    assert env.gold == pytest.approx(4000.0)
    assert env.merchant_stocks[MERCHANT_NAME][
        "Bethrezen's Claw (Artifact)"
    ] == 0
    assert env.equipped_artifact_items[0] == "Bethrezen's Claw (Artifact)"


def test_trade_train_elite_mercenary_roster_has_strong_front_and_back_units():
    env = _make_env()
    env.reset(seed=123)

    expected_names = [
        str(entry["unit_name"])
        for entry in ELITE_MERCENARIES
    ]
    assert [
        str(option["unit_name"])
        for option in env.active_mercenary_hire_options
    ] == expected_names
    assert [str(option["stand"]) for option in env.active_mercenary_hire_options] == [
        "ahead",
        "behind",
        "ahead",
        "behind",
        "ahead",
        "behind",
        "ahead",
        "behind",
    ]

    env.grid_env.agent_pos = env.mercenary_site_interaction_tiles[
        MERCENARY_CAMP_NAME
    ][0]
    hire_actions = [
        env.GRID_MERCENARY_HIRE_ACTION_START + idx
        for idx in range(len(ELITE_MERCENARIES))
    ]
    mask = env.compute_action_mask()
    assert all(bool(mask[action]) for action in hire_actions)

    _, _, terminated, truncated, info = env.step(hire_actions[0])

    assert terminated is False
    assert truncated is False
    assert info["hired"] is True
    assert info["hired_unit_name"] == "Защитник Веры"
    assert info["hired_position"] == 7
    assert info["mercenary_hire_site"] == MERCENARY_CAMP_NAME
    assert env.gold == pytest.approx(6500.0)


def test_trade_train_trainer_can_train_the_starting_duke():
    env = _make_env()
    env.reset(seed=123)
    env.grid_env.agent_pos = env.trainer_site_interaction_tiles[TRAINER_NAME][0]

    trainer_action = (
        env.GRID_TRAINER_ACTION_START
        + env.TRAINER_POSITIONS.index(8)
    )
    assert bool(env.compute_action_mask()[trainer_action]) is True

    _, _, terminated, truncated, info = env.step(trainer_action)

    assert terminated is False
    assert truncated is False
    assert info["trainer_trained"] is True
    assert info["trainer_unit_name"] == "Герцог"
    assert info["trainer_xp_gained"] > 0
    assert info["trainer_gold_spent"] > 0.0


def test_trade_train_has_four_racial_armies_and_two_orc_armies():
    env = _make_env()
    env.reset(seed=123)

    expected_compositions = {
        1: Counter(
            {
                "Имперский рыцарь": 1,
                "Рыцарь на пегасе": 1,
                "Патриарх": 1,
                "Волшебник": 1,
            }
        ),
        2: Counter(
            {
                "Кентавр дикарь": 1,
                "Кентавр латник": 1,
                "Солнечная танцовщица": 1,
                "Смотритель": 1,
            }
        ),
        3: Counter(
            {
                "Скелет рыцарь": 1,
                "Лорд тьмы": 1,
                "Архилич": 1,
                "Сущий": 1,
            }
        ),
        4: Counter(
            {
                "Король гномов": 1,
                "Старый ветеран": 1,
                "Архидруид": 1,
                "Огнеметчик": 1,
            }
        ),
        5: Counter({"Король орков": 1}),
        6: Counter({"Орк": 1, "Гоблин лучник": 1}),
    }

    assert len(ENEMY_STACKS) == 6
    for enemy_id, expected in expected_compositions.items():
        living_units = _living_units(env._enemy_configs[enemy_id])
        assert Counter(str(unit["name"]) for unit in living_units) == expected
        if enemy_id <= 4:
            assert sum(int(unit["position"]) in (1, 2, 3) for unit in living_units) == 2
            assert sum(int(unit["position"]) in (4, 5, 6) for unit in living_units) == 2

    orc_patrol = _living_units(env._enemy_configs[6])
    assert next(unit for unit in orc_patrol if unit["name"] == "Орк")["position"] == 2
    assert next(
        unit for unit in orc_patrol if unit["name"] == "Гоблин лучник"
    )["position"] == 5
