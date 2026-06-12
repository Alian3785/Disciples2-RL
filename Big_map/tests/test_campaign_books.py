import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv
from campaign_env import CampaignEnv


def _book_action(env: CampaignEnv, item_name: str) -> int:
    return env.GRID_EQUIP_BOOK_ACTION_START + env.scenario_book_item_names.index(item_name)


def _battle_equip_action(env: CampaignEnv, item_name: str) -> int:
    return env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(
        item_name
    )


class BookScenarioEnv(CampaignEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_chests = dict(self._static_chests)
        self._static_chests[(1, 1)] = (
            self.TOME_OF_WAR_ITEM_NAME,
            self.TOME_OF_SORCERY_ITEM_NAME,
            self.TOME_OF_FIRE_ITEM_NAME,
        )
        self.chests = dict(self._static_chests)
        self._refresh_dynamic_action_layout()
        self.action_space = self.action_space.__class__(
            self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT
        )


def _new_book_env() -> BookScenarioEnv:
    env = BookScenarioEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    return env


def _travel_hero(env: CampaignEnv) -> dict:
    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    return hero


def _grant_book_lore_level(env: CampaignEnv) -> dict:
    hero = _travel_hero(env)
    hero["Level"] = max(11, int(hero.get("Level", 0) or 0))
    return hero


def _grant_book_lore_token(env: CampaignEnv) -> dict:
    hero = _travel_hero(env)
    tokens = list(hero.get("hero_abilities") or [])
    if env.HERO_BOOK_LORE_ABILITY_KEY not in tokens:
        tokens.append(env.HERO_BOOK_LORE_ABILITY_KEY)
    hero["hero_abilities"] = tokens
    return hero


def test_book_actions_are_reserved_but_masked_until_owned():
    env = _new_book_env()
    _grant_book_lore_level(env)

    assert env.TOME_OF_ARCANUM_ITEM_NAME in env.scenario_book_item_names
    assert env.TOME_OF_SORCERY_ITEM_NAME in env.scenario_book_item_names
    assert env.grid_legion_damage_spell_action_start == (
        env.GRID_EQUIP_BOOK_ACTION_START + len(env.scenario_book_item_names)
    )

    arcanum_action = _book_action(env, env.TOME_OF_ARCANUM_ITEM_NAME)
    assert bool(env.compute_action_mask()[arcanum_action]) is False

    env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)
    env._add_hero_item(env.TOME_OF_ARCANUM_ITEM_NAME)

    assert env.equipped_book_items == [env.TOME_OF_WAR_ITEM_NAME]
    assert bool(env.compute_action_mask()[arcanum_action]) is False
    env._advance_turns(1)
    assert bool(env.compute_action_mask()[arcanum_action]) is True


def test_book_equip_requires_book_lore_prerequisite():
    env = _new_book_env()
    env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)

    war_action = _book_action(env, env.TOME_OF_WAR_ITEM_NAME)
    assert env.equipped_book_items == [None]
    assert bool(env.compute_action_mask()[war_action]) is False

    _, _, _, _, blocked_info = env.step(war_action)
    assert blocked_info["equip_book_applied"] is False
    assert env.equipped_book_items == [None]

    hero = _grant_book_lore_level(env)
    assert hero["Level"] == 11
    assert "Колдовство (Позволяет предводителю читать магические книги)" in (
        env.get_travel_hero_visual_info()["abilities"]
    )
    assert bool(env.compute_action_mask()[war_action]) is True


def test_book_pickup_auto_equips_only_when_slot_is_empty():
    env = _new_book_env()
    _grant_book_lore_level(env)

    first_info = env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)
    second_info = env._add_hero_item(env.TOME_OF_ARCANUM_ITEM_NAME)

    assert first_info["book_auto_equipped"] is True
    assert first_info["book_auto_equip_slot"] == 1
    assert second_info["book_auto_equipped"] is False
    assert second_info["book_auto_equip_previous"] == env.TOME_OF_WAR_ITEM_NAME
    assert env.equipped_book_items == [env.TOME_OF_WAR_ITEM_NAME]
    assert env.book_equip_used_this_turn is True
    assert env.books_equipped_total == 1


def test_book_slot_is_encoded_as_one_hot_resource_observation():
    env = _new_book_env()
    _grant_book_lore_level(env)
    book_offset = 11 + env.grid_battle_equipment_obs_size
    book_features = env.grid_book_equipment_features_per_slot

    empty_obs = env._build_resource_grid_obs()
    assert env.grid_book_equipment_item_names == env.BOOK_ITEM_NAMES
    assert empty_obs[book_offset] == pytest.approx(1.0)
    assert sum(empty_obs[book_offset : book_offset + book_features]) == pytest.approx(1.0)

    env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)
    war_index = env.grid_book_equipment_item_names.index(env.TOME_OF_WAR_ITEM_NAME)
    war_obs = env._build_resource_grid_obs()

    assert war_obs[book_offset] == pytest.approx(0.0)
    assert war_obs[book_offset + 1 + war_index] == pytest.approx(1.0)
    assert sum(war_obs[book_offset : book_offset + book_features]) == pytest.approx(1.0)


def test_book_observation_size_uses_fixed_book_vocabulary():
    base_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    base_env.reset(seed=123)
    book_env = _new_book_env()

    assert base_env.scenario_book_item_names != book_env.scenario_book_item_names
    assert base_env.grid_book_equipment_item_names == base_env.BOOK_ITEM_NAMES
    assert book_env.grid_book_equipment_item_names == book_env.BOOK_ITEM_NAMES
    assert base_env.grid_book_equipment_obs_size == book_env.grid_book_equipment_obs_size
    assert base_env.grid_resource_obs_size == book_env.grid_resource_obs_size
    assert base_env.observation_space.shape == book_env.observation_space.shape


def test_manual_book_replacement_has_own_turn_lock_and_does_not_block_battle_items():
    env = _new_book_env()
    _grant_book_lore_level(env)
    env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)
    env._add_hero_item(env.TOME_OF_ARCANUM_ITEM_NAME)
    env._add_hero_item(env.TOME_OF_SORCERY_ITEM_NAME)
    env._append_hero_item(env.HEALING_OINTMENT_ITEM_NAME)
    env._advance_turns(1)

    arcanum_action = _book_action(env, env.TOME_OF_ARCANUM_ITEM_NAME)
    sorcery_action = _book_action(env, env.TOME_OF_SORCERY_ITEM_NAME)
    battle_item_action = _battle_equip_action(env, env.HEALING_OINTMENT_ITEM_NAME)

    _, _, _, _, book_info = env.step(arcanum_action)

    assert book_info["equip_book_applied"] is True
    assert book_info["equip_book_previous"] == env.TOME_OF_WAR_ITEM_NAME
    assert env.equipped_book_items == [env.TOME_OF_ARCANUM_ITEM_NAME]
    assert bool(env.compute_action_mask()[sorcery_action]) is False
    assert bool(env.compute_action_mask()[battle_item_action]) is True

    _, _, _, _, battle_info = env.step(battle_item_action)

    assert battle_info["equip_battle_item_applied"] is True
    assert env.equipped_hero_items[0] == env.HEALING_OINTMENT_ITEM_NAME

    env._advance_turns(1)
    assert bool(env.compute_action_mask()[sorcery_action]) is True


def test_arcanum_and_sorcery_unlock_only_their_battle_item_type_without_sorcery_lore():
    env = _new_book_env()
    hero = _travel_hero(env)
    hero["Level"] = 1
    hero["hero_abilities"] = []
    _grant_book_lore_token(env)
    env.typeoflord = 1
    env._append_hero_item("Vampire Orb")
    env._append_hero_item("Zombie Talisman")

    orb_action = _battle_equip_action(env, "Vampire Orb")
    talisman_action = _battle_equip_action(env, "Zombie Talisman")
    assert bool(env.compute_action_mask()[orb_action]) is False
    assert bool(env.compute_action_mask()[talisman_action]) is False

    env._add_hero_item(env.TOME_OF_ARCANUM_ITEM_NAME)
    assert env.equipped_book_items == [env.TOME_OF_ARCANUM_ITEM_NAME]
    assert bool(env.compute_action_mask()[orb_action]) is True
    assert bool(env.compute_action_mask()[talisman_action]) is False

    _, _, _, _, orb_info = env.step(orb_action)
    assert orb_info["equip_battle_item_applied"] is True
    assert env.equipped_hero_items[0] == "Vampire Orb"

    env._add_hero_item(env.TOME_OF_SORCERY_ITEM_NAME)
    env._advance_turns(1)
    _, _, _, _, book_info = env.step(_book_action(env, env.TOME_OF_SORCERY_ITEM_NAME))

    assert book_info["equip_book_applied"] is True
    assert env.equipped_book_items == [env.TOME_OF_SORCERY_ITEM_NAME]
    assert env.equipped_hero_items[0] is None
    assert bool(env.compute_action_mask()[orb_action]) is False
    assert bool(env.compute_action_mask()[talisman_action]) is True


def test_protective_book_grants_single_resistance_to_hero_for_battle_only():
    env = _new_book_env()
    _grant_book_lore_level(env)
    env._add_hero_item(env.TOME_OF_FIRE_ITEM_NAME)
    before = deepcopy(_travel_hero(env))

    env._init_battle(enemy_id=1)
    assert env.battle_env is not None
    battle_hero = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and bool(unit.get("hero"))
    )

    assert "Fire" in set(battle_hero.get("resistance") or [])
    attacker = {"attack_type_primary": "Fire", "team": "red", "name": "Fire", "position": 1}
    assert env.battle_env._resilience_blocks(attacker, battle_hero) is True
    assert env.battle_env._resilience_blocks(attacker, battle_hero) is False

    env._save_blue_state()
    saved_hero = _travel_hero(env)
    assert list(saved_hero.get("resistance") or []) == list(before.get("resistance") or [])
    assert "campaign_book_added_resistance" not in saved_hero
    assert "campaign_active_book" not in saved_hero


def test_tome_of_war_increases_distributed_battle_exp():
    env = _new_book_env()
    _grant_book_lore_level(env)
    env._add_hero_item(env.TOME_OF_WAR_ITEM_NAME)
    env._init_battle(enemy_id=1)
    assert env.battle_env is not None
    assert env.battle_env.exp_multiplier == pytest.approx(1.25)

    battle = BattleEnv(log_enabled=False)
    battle.exp_multiplier = env._active_book_exp_multiplier()
    blue = {
        "team": "blue",
        "name": "Hero",
        "position": 8,
        "health": 100,
        "running_away": 0,
        "exp_current": 0,
        "exp_required": 999,
    }
    red = {
        "team": "red",
        "name": "Target",
        "position": 1,
        "health": 0,
        "running_away": 0,
        "exp_kill": 40,
    }
    battle.combined = [blue, red]

    battle._apply_battle_exp("red")

    assert battle.last_battle_exp == pytest.approx(50.0)
    assert blue["exp_current"] == pytest.approx(50.0)

    red_wins_battle = BattleEnv(log_enabled=False)
    red_wins_battle.exp_multiplier = env._active_book_exp_multiplier()
    losing_blue = {
        "team": "blue",
        "name": "Hero",
        "position": 8,
        "health": 0,
        "running_away": 0,
        "exp_kill": 40,
    }
    winning_red = {
        "team": "red",
        "name": "Target",
        "position": 1,
        "health": 100,
        "running_away": 0,
        "exp_current": 0,
        "exp_required": 999,
    }
    red_wins_battle.combined = [losing_blue, winning_red]

    red_wins_battle._apply_battle_exp("blue")

    assert red_wins_battle.last_battle_exp == pytest.approx(40.0)
    assert winning_red["exp_current"] == pytest.approx(40.0)
