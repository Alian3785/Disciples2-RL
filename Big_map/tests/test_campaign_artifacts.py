import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


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


def test_artifacts_auto_equip_into_two_rotating_slots():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    ring_info = env._grant_ruin_reward(74)
    assert ring_info["artifact_auto_equipped"] is True
    assert ring_info["artifact_auto_equip_slot"] == 1
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]

    env._add_hero_item("Holy Chalice (Artifact)")
    assert env.equipped_artifact_items == [
        env.RING_OF_AGES_ITEM_NAME,
        "Holy Chalice (Artifact)",
    ]

    env._add_hero_item("Unholy Chalice (Artifact)")
    assert env.equipped_artifact_items == [
        "Unholy Chalice (Artifact)",
        "Holy Chalice (Artifact)",
    ]

    env._add_hero_item("Runestone (Artifact)")
    assert env.equipped_artifact_items == [
        "Unholy Chalice (Artifact)",
        "Runestone (Artifact)",
    ]

    env._add_hero_item("Talisman of Ages (Artifact)")
    assert env.equipped_artifact_items == [
        "Talisman of Ages (Artifact)",
        "Runestone (Artifact)",
    ]
    assert env.artifact_auto_equip_next_slot == 1


def test_ring_of_ages_buffs_only_hero_and_restores_after_battle_save():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

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

    assert hero_battle["name"] == hero_before["name"]
    assert hero_battle["damage"] == expected_damage
    assert hero_battle["initiative_base"] == expected_initiative
    assert hero_battle["campaign_active_artifacts"] == [env.RING_OF_AGES_ITEM_NAME]

    assert ally_battle["damage"] == ally_before["damage"]
    assert ally_battle["initiative_base"] == ally_before["initiative_base"]

    env._save_blue_state()

    hero_after = env._resolve_travel_hero(units=env.blue_team_state)
    ally_after = _blue_state_unit(env, 7)

    assert hero_after["damage"] == hero_before["damage"]
    assert hero_after["initiative_base"] == hero_before["initiative_base"]
    assert "campaign_active_artifacts" not in hero_after
    assert ally_after["damage"] == ally_before["damage"]
    assert ally_after["initiative_base"] == ally_before["initiative_base"]
    assert env.equipped_artifact_items == [env.RING_OF_AGES_ITEM_NAME, None]


def test_render_and_state_summary_include_equipped_artifacts(capsys: pytest.CaptureFixture[str]):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

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
