import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv


def _make_enemy(exp_kill: int) -> dict:
    return {
        "name": "Test enemy",
        "team": "red",
        "position": 1,
        "health": 1,
        "max_health": 1,
        "running_away": 0,
        "exp_kill": exp_kill,
    }


def _set_lord_type(env: CampaignEnv, lord_type: int) -> None:
    env.typeoflord = int(lord_type)


def test_hero_levelup_raises_level_resets_exp_and_buffs_hero_stats():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 149
    duke["exp_required"] = 150
    duke["Level"] = 0
    duke["next_level_exp"] = 500
    duke["damage"] = 50
    duke["original_damage"] = 50
    duke["health"] = 150
    duke["max_health"] = 150
    duke["exp_kill"] = 60
    duke["accuracy"] = 80
    duke["needaunit"] = 0

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 1
    assert duke["exp_current"] == 0
    assert duke["exp_required"] == 650
    assert duke["damage"] == 55
    assert duke["original_damage"] == 55
    assert duke["health"] == 165
    assert duke["max_health"] == 165
    assert duke["exp_kill"] == 66
    assert duke["accuracy"] == 81
    assert duke["needaunit"] == 0
    assert env.last_levelups == [duke["name"]]


def test_hero_levelup_to_third_level_sets_needaunit_flag():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 649
    duke["exp_required"] = 650
    duke["Level"] = 2
    duke["next_level_exp"] = 500
    duke["needaunit"] = 0

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 3
    assert duke["needaunit"] == 1


def test_hero_levelup_to_fourth_level_grants_endurance_health_bonus():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 1149
    duke["exp_required"] = 1150
    duke["Level"] = 3
    duke["next_level_exp"] = 500
    duke["health"] = 200
    duke["max_health"] = 200
    duke["exp_kill"] = 73
    duke["accuracy"] = 82
    duke["needaunit"] = 0

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 4
    assert duke["exp_current"] == 0
    assert duke["exp_required"] == 1650
    assert duke["health"] == 264
    assert duke["max_health"] == 264
    assert duke["exp_kill"] == 80
    assert duke["accuracy"] == 83
    assert duke["needaunit"] == 0


def test_regular_unit_without_next_level_exp_keeps_transform_upgrade_flow():
    env = BattleEnv(log_enabled=False)
    fighter = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 7))
    fighter["exp_current"] = 94
    fighter["exp_required"] = 95
    fighter["next_level_exp"] = 0
    initial_level = fighter.get("Level", 0)

    env.combined = [fighter, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert fighter["Level"] == initial_level
    assert fighter["exp_current"] == 94
    assert fighter["exp_required"] == 95
    assert env.last_levelups == [fighter["name"]]


def test_campaign_save_preserves_hero_level_and_exp_threshold_after_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    battle_units = deepcopy(UNITS_BLUE)
    duke = next(u for u in battle_units if int(u.get("position", -1)) == 8)
    duke["Level"] = 1
    duke["exp_required"] = 650
    duke["exp_current"] = 0
    duke["next_level_exp"] = 500

    env.battle_env = type("DummyBattle", (), {"combined": battle_units})()
    env._save_blue_state()

    saved_duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    assert saved_duke["Level"] == 1
    assert saved_duke["exp_required"] == 650
    assert saved_duke["exp_current"] == 0
    assert saved_duke["next_level_exp"] == 500
    assert env.moves_per_turn == 20
    assert env.moves == 20


def test_campaign_moves_per_turn_scales_with_travel_hero_level():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    assert duke["Level"] == 1
    assert duke["needaunit"] == 0
    assert env.moves_per_turn == 20
    env.moves = 5
    duke["Level"] = 2

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.moves_per_turn == 25
    assert env.moves == 10
    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 2,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ],
    }

    env._advance_turns(1)
    assert env.moves == 25


def test_campaign_base_moves_per_turn_uses_alive_hero_type():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    hero = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    hero["name"] = "Советник"
    hero["Level"] = 1
    hero["health"] = 150
    hero["hp"] = 150

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)

    assert env.moves_per_turn == 35
    assert env.moves == 35

    hero["health"] = 0
    hero["hp"] = 0
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, refill=True)

    assert env.moves_per_turn == 20
    assert env.moves == 20


def test_campaign_travel_hero_visual_info_shows_leadership_on_level_three():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 3
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 3,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Лидерство",
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ],
    }


def test_campaign_travel_hero_visual_info_shows_endurance_on_level_four():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 4
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 4,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Лидерство",
            "Выносливость",
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ],
    }


def test_lord_types_grant_their_starting_hero_knowledge():
    cases = [
        (
            1,
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ),
        (
            2,
            "Колдовские знания (Позволяет предводителю экипировать и использовать сферы в бою)",
        ),
        (
            3,
            "Походные знания (Позволяет предводителю носить магические ботинки)",
        ),
    ]
    for lord_type, expected_ability in cases:
        env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
        env.reset(seed=123)
        _set_lord_type(env, lord_type)

        info = env.get_travel_hero_visual_info()

        assert info["level"] == 1
        assert info["abilities"] == [expected_ability]


def test_campaign_travel_hero_visual_info_shows_banner_bearer_on_level_seven():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 7
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 7,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Лидерство x2",
            "Выносливость",
            "Сила (+25% основной урон)",
            "Знаменосец (Позволяет предводителю использовать знамёна)",
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ],
    }


def test_campaign_travel_hero_visual_info_shows_marching_lore_on_level_eight():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 8
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 8,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Лидерство x2",
            "Выносливость",
            "Сила (+25% основной урон)",
            "Знаменосец (Позволяет предводителю использовать знамёна)",
            "Походные знания (Позволяет предводителю носить магические ботинки)",
            "Знание артефактов (Позволяет предводителю использовать артефакты.)",
        ],
    }


def test_campaign_travel_hero_visual_info_shows_artifact_knowledge_on_level_nine():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 9
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    info = env.get_travel_hero_visual_info()
    assert info["name"] == duke["name"]
    assert info["level"] == 9
    assert (
        "Знание артефактов (Позволяет предводителю использовать артефакты.)"
        in info["abilities"]
    )
    assert info["abilities"][-1] == (
        "Мощь (+25% основной урон героя)"
    )


def test_campaign_travel_hero_visual_info_shows_sorcery_lore_on_level_ten():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 10
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    info = env.get_travel_hero_visual_info()
    assert info["name"] == duke["name"]
    assert info["level"] == 10
    assert (
        "Колдовские знания (Позволяет предводителю экипировать и использовать сферы в бою)"
        in info["abilities"]
    )
    assert info["abilities"][-1] == (
        "Мощь (+25% основной урон героя)"
    )


def test_campaign_travel_hero_visual_info_shows_book_lore_on_level_eleven():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 11
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    info = env.get_travel_hero_visual_info()
    assert info["name"] == duke["name"]
    assert info["level"] == 11
    assert (
        "Колдовство (Позволяет предводителю читать магические книги)"
        in info["abilities"]
    )
    assert info["abilities"][-1] == (
        "Мощь (+25% основной урон героя)"
    )


def test_replacement_might_bonus_applies_once_and_only_to_primary_damage():
    cases = [
        (1, 9),
        (2, 10),
        (3, 8),
    ]
    for lord_type, replacement_level in cases:
        env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
        env.reset(seed=123)
        _set_lord_type(env, lord_type)

        battle_units = deepcopy(UNITS_BLUE)
        duke = next(u for u in battle_units if int(u.get("position", -1)) == 8)
        duke["Level"] = replacement_level
        duke["damage"] = 100
        duke["original_damage"] = 100
        duke["damage_secondary"] = 17
        duke["health"] = 150
        duke["hp"] = 150
        duke["max_health"] = 150
        duke["maxhp"] = 150

        env.battle_env = type("DummyBattle", (), {"combined": battle_units})()
        env._save_blue_state()

        saved_duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
        assert saved_duke["damage"] == 125
        assert saved_duke["original_damage"] == 125
        assert saved_duke["damage_secondary"] == 17
        assert saved_duke["campaign_lord_might_bonus_levels"] == [replacement_level]
        assert "Мощь (+25% основной урон героя)" in env.get_travel_hero_visual_info()["abilities"]

        env.battle_env = type(
            "DummyBattle",
            (),
            {"combined": deepcopy(env.blue_team_state)},
        )()
        env._save_blue_state()

        saved_again = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
        assert saved_again["damage"] == 125
        assert saved_again.get("original_damage", saved_again["damage"]) == 125
        assert saved_again["damage_secondary"] == 17
