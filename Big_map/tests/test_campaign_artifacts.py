import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _set_hero_level(env: CampaignEnv, level: int) -> dict:
    env.typeoflord = 2
    hero = env._resolve_travel_hero()
    assert hero is not None
    hero["Level"] = int(level)
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)
    return hero


def _blue_state_unit(env: CampaignEnv, position: int) -> dict:
    roster = env.blue_team_state if env.blue_team_state is not None else env._get_blue_state()
    return next(
        unit for unit in roster if int(unit.get("position", -1) or -1) == int(position)
    )


def _battle_blue_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(position)
    )


def _battle_blue_hero(env: CampaignEnv) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and bool(unit.get("hero"))
    )


def _battle_red_target(env: CampaignEnv) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "red" and int(unit.get("health", 0) or 0) > 0
    )


def _prepare_artifact_attack(item_name: str):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)
    env._add_hero_item(item_name)
    env._init_battle(enemy_id=1)

    assert env.battle_env is not None
    battle = env.battle_env
    status_chances = []
    battle._roll_hit = lambda attacker: True
    battle._roll_status = lambda chance: status_chances.append(chance) or True
    battle._apply_damage_with_armor = lambda attacker, base_dmg, victim: 1

    hero = _battle_blue_hero(env)
    target = _battle_red_target(env)
    hero["damage"] = 1
    hero["attack_type_primary"] = "Weapon"
    target["name"] = "Artifact Target"
    target["health"] = 1000
    target["max_health"] = 1000
    target["immunity"] = []
    target["resistance"] = []
    target["resilience_used_types"] = []
    target["paralyzed"] = 0
    target["long_paralyzed"] = 0
    target["poison_turns_left"] = 0
    target["poison_damage_per_tick"] = 0
    return env, hero, target, status_chances


def test_artifacts_without_artifact_knowledge_stay_in_inventory_without_effects():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.typeoflord = 2
    _set_hero_level(env, 8)
    hero_before = deepcopy(env._resolve_travel_hero())

    equip_info = env._add_hero_item(env.RING_OF_AGES_ITEM_NAME)

    hero_after = env._resolve_travel_hero()
    assert equip_info["artifact_auto_equipped"] is False
    assert equip_info["artifact_auto_equip_slot"] is None
    assert env.equipped_artifact_items == [None, None]
    assert env.RING_OF_AGES_ITEM_NAME in [env._hero_item_name(entry) for entry in env.heroitems]
    assert hero_after["damage"] == hero_before["damage"]
    assert hero_after["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_after["initiative_base"] == hero_before["initiative_base"]
    assert "campaign_active_artifacts" not in hero_after


def test_warrior_lord_can_use_artifacts_from_level_one():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    hero_before = deepcopy(env._resolve_travel_hero())

    equip_info = env._add_hero_item(env.RING_OF_AGES_ITEM_NAME)

    expected_damage = env._normalize_damage_value(
        float(hero_before["damage"]) * float(env.RING_OF_AGES_DAMAGE_MULTIPLIER)
    )
    hero_after = env._resolve_travel_hero()
    assert equip_info["artifact_auto_equipped"] is True
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]
    assert hero_after["damage"] == expected_damage
    assert hero_after["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_after["campaign_active_artifacts"] == [env.RING_OF_AGES_ITEM_NAME]


def test_artifact_knowledge_ability_token_unlocks_artifacts_below_level_nine():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    hero = _set_hero_level(env, 8)
    hero["hero_abilities"] = ["artifact_knowledge"]

    equip_info = env._add_hero_item(env.RING_OF_AGES_ITEM_NAME)

    assert equip_info["artifact_auto_equipped"] is True
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]
    assert "Знание артефактов" in " ".join(env.get_travel_hero_visual_info()["abilities"])


def test_artifacts_auto_equip_best_two_by_gold_value():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)

    ring_info = env._grant_ruin_reward(74)
    assert ring_info["artifact_auto_equipped"] is True
    assert ring_info["artifact_auto_equip_slot"] == 1
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]

    holy_info = env._add_hero_item("Holy Chalice (Artifact)")
    assert holy_info["artifact_auto_equipped"] is True
    assert holy_info["artifact_auto_equip_slot"] == 2
    assert env.equipped_artifact_items == [
        env.RING_OF_AGES_ITEM_NAME,
        "Holy Chalice (Artifact)",
    ]

    unholy_info = env._add_hero_item("Unholy Chalice (Artifact)")
    assert unholy_info["artifact_auto_equipped"] is False
    assert unholy_info["artifact_auto_equip_slot"] is None
    assert env.equipped_artifact_items == [
        env.RING_OF_AGES_ITEM_NAME,
        "Holy Chalice (Artifact)",
    ]

    bethrezen_info = env._add_hero_item(env.BETHREZENS_CLAW_ITEM_NAME)
    assert bethrezen_info["artifact_auto_equipped"] is True
    assert bethrezen_info["artifact_auto_equip_slot"] == 1
    assert bethrezen_info["artifact_auto_equip_previous"] == env.RING_OF_AGES_ITEM_NAME
    assert env.equipped_artifact_items == [
        env.BETHREZENS_CLAW_ITEM_NAME,
        env.RING_OF_AGES_ITEM_NAME,
    ]

    skull_info = env._add_hero_item(env.SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME)
    assert skull_info["artifact_auto_equipped"] is False
    assert skull_info["artifact_auto_equip_slot"] is None
    assert env.equipped_artifact_items == [
        env.BETHREZENS_CLAW_ITEM_NAME,
        env.RING_OF_AGES_ITEM_NAME,
    ]
    assert env.artifact_auto_equip_next_slot == 0


def test_ring_of_ages_buffs_only_hero_and_persists_after_battle_save():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)

    hero_before = deepcopy(env._resolve_travel_hero())
    ally_before = deepcopy(_blue_state_unit(env, 7))

    env._grant_ruin_reward(74)
    env._init_battle(enemy_id=1)

    hero_battle = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and bool(unit.get("hero"))
    )
    ally_battle = _battle_blue_unit(env, 7)

    expected_damage = env._normalize_damage_value(
        float(hero_before["damage"]) * float(env.RING_OF_AGES_DAMAGE_MULTIPLIER)
    )
    expected_initiative = env._normalize_damage_value(
        float(hero_before["initiative_base"]) * float(env.RING_OF_AGES_INITIATIVE_MULTIPLIER)
    )

    hero_campaign = env._resolve_travel_hero()
    assert hero_campaign["damage"] == expected_damage
    assert hero_campaign["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_campaign["initiative_base"] == expected_initiative
    assert hero_campaign["campaign_active_artifacts"] == [env.RING_OF_AGES_ITEM_NAME]

    assert hero_battle["name"] == hero_before["name"]
    assert hero_battle["damage"] == expected_damage
    assert hero_battle["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_battle["initiative_base"] == expected_initiative
    assert hero_battle["campaign_active_artifacts"] == [env.RING_OF_AGES_ITEM_NAME]

    assert ally_battle["damage"] == ally_before["damage"]
    assert ally_battle["damage_secondary"] == ally_before["damage_secondary"]
    assert ally_battle["initiative_base"] == ally_before["initiative_base"]

    env._save_blue_state()

    hero_after = env._resolve_travel_hero(units=env.blue_team_state)
    ally_after = _blue_state_unit(env, 7)

    assert hero_after["damage"] == expected_damage
    assert hero_after["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_after["initiative_base"] == expected_initiative
    assert hero_after["campaign_active_artifacts"] == [env.RING_OF_AGES_ITEM_NAME]
    assert ally_after["damage"] == ally_before["damage"]
    assert ally_after["damage_secondary"] == ally_before["damage_secondary"]
    assert ally_after["initiative_base"] == ally_before["initiative_base"]
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]

    env._add_hero_item("Holy Chalice (Artifact)")
    env._add_hero_item("Runestone (Artifact)")
    assert env.equipped_artifact_items == [
        env.RING_OF_AGES_ITEM_NAME,
        "Holy Chalice (Artifact)",
    ]
    assert env._consume_hero_item(env.RING_OF_AGES_ITEM_NAME) is True

    hero_unequipped = env._resolve_travel_hero()
    assert env.equipped_artifact_items == ["Holy Chalice (Artifact)", "Runestone (Artifact)"]
    assert hero_unequipped["damage"] == hero_before["damage"]
    assert hero_unequipped["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_unequipped["initiative_base"] == hero_before["initiative_base"]
    assert hero_unequipped["armor"] == (
        int(hero_before.get("armor", 0) or 0)
        + env.RUNESTONE_ARTIFACT_ARMOR_BONUS
        + env.HOLY_CHALICE_ARTIFACT_ARMOR_BONUS
    )
    assert "campaign_artifact_base_damage" not in hero_unequipped
    assert "campaign_artifact_base_initiative" not in hero_unequipped


@pytest.mark.parametrize(
    ("item_name", "damage_multiplier", "initiative_multiplier"),
    [
        (
            CampaignEnv.DWARVEN_BRACER_ITEM_NAME,
            CampaignEnv.DWARVEN_BRACER_DAMAGE_MULTIPLIER,
            1.0,
        ),
        (
            CampaignEnv.UNHOLY_CHALICE_ARTIFACT_ITEM_NAME,
            CampaignEnv.UNHOLY_CHALICE_ARTIFACT_DAMAGE_MULTIPLIER,
            1.0,
        ),
        (
            CampaignEnv.RING_OF_STRENGTH_ITEM_NAME,
            CampaignEnv.RING_OF_STRENGTH_DAMAGE_MULTIPLIER,
            1.0,
        ),
        (
            CampaignEnv.RUNIC_BLADE_ITEM_NAME,
            CampaignEnv.RUNIC_BLADE_DAMAGE_MULTIPLIER,
            1.0,
        ),
        (
            CampaignEnv.MJOLNIRS_CROWN_ITEM_NAME,
            CampaignEnv.MJOLNIRS_CROWN_DAMAGE_MULTIPLIER,
            1.0,
        ),
        (
            CampaignEnv.BETHREZENS_CLAW_ITEM_NAME,
            CampaignEnv.BETHREZENS_CLAW_DAMAGE_MULTIPLIER,
            CampaignEnv.BETHREZENS_CLAW_INITIATIVE_MULTIPLIER,
        ),
    ],
)
def test_weapon_artifacts_buff_equipped_hero_in_campaign_and_battle(
    item_name: str,
    damage_multiplier: float,
    initiative_multiplier: float,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)

    hero = env._resolve_travel_hero()
    hero["damage_secondary"] = 13
    hero_before = deepcopy(hero)
    ally_before = deepcopy(_blue_state_unit(env, 7))

    equip_info = env._add_hero_item(item_name)

    expected_damage = env._normalize_damage_value(
        float(hero_before["damage"]) * float(damage_multiplier)
    )
    expected_initiative = env._normalize_damage_value(
        float(hero_before["initiative_base"]) * float(initiative_multiplier)
    )

    hero_campaign = env._resolve_travel_hero()
    assert equip_info["artifact_auto_equipped"] is True
    assert env.equipped_artifact_items == [item_name, None]
    assert hero_campaign["damage"] == expected_damage
    assert hero_campaign["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_campaign["initiative_base"] == expected_initiative
    assert hero_campaign["campaign_active_artifacts"] == [item_name]

    env._init_battle(enemy_id=1)

    hero_battle = _battle_blue_hero(env)
    ally_battle = _battle_blue_unit(env, 7)
    assert hero_battle["damage"] == expected_damage
    assert hero_battle["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_battle["initiative_base"] == expected_initiative
    assert hero_battle["campaign_active_artifacts"] == [item_name]
    assert ally_battle["damage"] == ally_before["damage"]
    assert ally_battle["damage_secondary"] == ally_before["damage_secondary"]
    assert ally_battle["initiative_base"] == ally_before["initiative_base"]

    env._save_blue_state()

    hero_after = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero_after["damage"] == expected_damage
    assert hero_after["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_after["initiative_base"] == expected_initiative
    assert hero_after["campaign_active_artifacts"] == [item_name]


@pytest.mark.parametrize(
    ("item_name", "armor_bonus"),
    [
        (CampaignEnv.RUNESTONE_ARTIFACT_ITEM_NAME, CampaignEnv.RUNESTONE_ARTIFACT_ARMOR_BONUS),
        (
            CampaignEnv.HOLY_CHALICE_ARTIFACT_ITEM_NAME,
            CampaignEnv.HOLY_CHALICE_ARTIFACT_ARMOR_BONUS,
        ),
        (CampaignEnv.SKULL_BRACERS_ITEM_NAME, CampaignEnv.SKULL_BRACERS_ARMOR_BONUS),
        (CampaignEnv.HORN_OF_AWARENESS_ITEM_NAME, CampaignEnv.HORN_OF_AWARENESS_ARMOR_BONUS),
        (CampaignEnv.ETCHED_CIRCLET_ITEM_NAME, CampaignEnv.ETCHED_CIRCLET_ARMOR_BONUS),
    ],
)
def test_armor_artifacts_buff_hero_armor_in_campaign_and_battle(
    item_name: str,
    armor_bonus: int,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)

    hero = env._resolve_travel_hero()
    hero["damage_secondary"] = 13
    hero_before = deepcopy(hero)
    ally_before = deepcopy(_blue_state_unit(env, 7))

    equip_info = env._add_hero_item(item_name)
    expected_armor = int(hero_before.get("armor", 0) or 0) + int(armor_bonus)

    hero_campaign = env._resolve_travel_hero()
    assert equip_info["artifact_auto_equipped"] is True
    assert env.equipped_artifact_items == [item_name, None]
    assert hero_campaign["armor"] == expected_armor
    assert hero_campaign["damage"] == hero_before["damage"]
    assert hero_campaign["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_campaign["initiative_base"] == hero_before["initiative_base"]
    assert hero_campaign["campaign_active_artifacts"] == [item_name]

    env._init_battle(enemy_id=1)

    hero_battle = _battle_blue_hero(env)
    ally_battle = _battle_blue_unit(env, 7)
    assert hero_battle["armor"] == expected_armor
    assert hero_battle["damage"] == hero_before["damage"]
    assert hero_battle["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_battle["initiative_base"] == hero_before["initiative_base"]
    assert hero_battle["campaign_active_artifacts"] == [item_name]
    assert ally_battle["armor"] == ally_before["armor"]
    assert ally_battle["damage"] == ally_before["damage"]
    assert ally_battle["damage_secondary"] == ally_before["damage_secondary"]
    assert ally_battle["initiative_base"] == ally_before["initiative_base"]

    env._save_blue_state()

    hero_after = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero_after["armor"] == expected_armor
    assert hero_after["damage"] == hero_before["damage"]
    assert hero_after["damage_secondary"] == hero_before["damage_secondary"]
    assert hero_after["initiative_base"] == hero_before["initiative_base"]
    assert hero_after["campaign_active_artifacts"] == [item_name]


def test_render_and_state_summary_include_equipped_artifacts(capsys: pytest.CaptureFixture[str]):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)

    env._add_hero_item("Holy Chalice (Artifact)")
    env._add_hero_item("Runestone (Artifact)")

    rendered = env.render(mode="ansi")
    captured = capsys.readouterr()

    assert rendered is not None
    assert "Экипированные артефакты: [Holy Chalice (Artifact), Runestone (Artifact)]" in rendered
    assert "Экипированные артефакты" in captured.out

    summary = env.get_state_summary()
    assert summary["equipped_artifact_items"] == [
        "Holy Chalice (Artifact)",
        "Runestone (Artifact)",
    ]
    assert summary["artifact_auto_equip_next_slot"] == 0


def test_soul_crystal_paralyzes_with_mind_chance():
    env, hero, target, status_chances = _prepare_artifact_attack(
        CampaignEnv.SOUL_CRYSTAL_ARTIFACT_ITEM_NAME
    )

    env.battle_env._attack(hero, target["position"])

    assert status_chances == [80]
    assert target["paralyzed"] == 1


def test_horn_of_incubus_short_paralyzes_with_earth_chance():
    env, hero, target, status_chances = _prepare_artifact_attack(
        CampaignEnv.HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME
    )

    env.battle_env._attack(hero, target["position"])

    assert status_chances == [80]
    assert target["paralyzed"] == 1
    assert target["long_paralyzed"] == 0


def test_horn_of_incubus_uses_earth_resilience():
    env, hero, target, status_chances = _prepare_artifact_attack(
        CampaignEnv.HORN_OF_INCUBUS_ARTIFACT_ITEM_NAME
    )
    target["resistance"] = ["Earth"]

    env.battle_env._attack(hero, target["position"])

    assert status_chances == [80]
    assert target["paralyzed"] == 0
    assert target["resilience_used_types"] == ["Earth"]


def test_unholy_dagger_drains_quarter_of_inflicted_damage():
    env, hero, target, status_chances = _prepare_artifact_attack(
        CampaignEnv.UNHOLY_DAGGER_ARTIFACT_ITEM_NAME
    )
    hero["health"] = 50
    hero["max_health"] = 100
    env.battle_env._apply_damage_with_armor = lambda attacker, base_dmg, victim: 40

    env.battle_env._attack(hero, target["position"])

    assert status_chances == []
    assert hero["health"] == 60


@pytest.mark.parametrize(
    ("item_name", "expected_damage"),
    [
        (CampaignEnv.THANATOS_BLADE_ARTIFACT_ITEM_NAME, 20),
        (CampaignEnv.SKULL_OF_THANATOS_ARTIFACT_ITEM_NAME, 35),
    ],
)
def test_poison_artifacts_apply_death_poison(item_name: str, expected_damage: int):
    env, hero, target, status_chances = _prepare_artifact_attack(item_name)

    env.battle_env._attack(hero, target["position"])

    assert status_chances == [80]
    assert target["poison_damage_per_tick"] == expected_damage
    assert 1 <= target["poison_turns_left"] <= 6


def test_hags_ring_transforms_with_mind_chance():
    env, hero, target, status_chances = _prepare_artifact_attack(
        CampaignEnv.HAGS_RING_ARTIFACT_ITEM_NAME
    )
    target["unit_type"] = "Mage"
    target["transformed"] = 0

    env.battle_env._attack(hero, target["position"])

    assert status_chances == [70]
    assert target["transformed"] == 1
    assert target["unit_type"] == "Warrior"
