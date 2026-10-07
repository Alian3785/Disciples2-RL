"""Orbs and talismans follow their Gattacks reach, accuracy, armor and revive share."""

import random

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


UNITS = {unit["кто"]: unit for unit in DATA}
EFFECTS = CampaignEnv._battle_item_effect_definitions()
MASS_ORBS = ("Fire", "Inferno", "Lightning", "Earth", "Stone Rain", "Water", "Icefall",
             "Nosferat", "Vampire", "Elder Vampires")


class FixedRng(random.Random):
    """random() drives the hit check; randint() returns the lowest bonus."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value

    def randint(self, low, high):
        return low


def battle(rng_value=0.0, enemies=(1, 2, 3)):
    red = [map_unit_to_battle(UNITS["Рыцарь"], "red", pos) for pos in enemies]
    for unit in red:
        unit.update(initiative_base=0, initiative=0, immunity=[], resistance=[])
    red += [placeholder_unit("red", pos) for pos in range(1, 7) if pos not in enemies]
    blue = [map_unit_to_battle(UNITS["Одержимый"], "blue", 7),
            map_unit_to_battle(UNITS["Герцог"], "blue", 8),
            map_unit_to_battle(UNITS["Сектант"], "blue", 11)]
    for unit in blue:
        unit.update(initiative_base=1000, initiative=1000)
    blue += [placeholder_unit("blue", pos) for pos in (9, 10, 12)]
    env = BattleEnv(log_enabled=False)
    env.hero_item_effects = EFFECTS
    env.rng.seed(1)
    env._init_with_custom_teams(red, blue)
    env.rng = FixedRng(rng_value)
    env.current_blue_attacker_pos = 8
    for pos in enemies:
        env._unit_by_position(pos).update(armor=40, health=1000, hp=1000, max_health=1000)
    return env


@pytest.mark.parametrize("kind", ["Orb", "Talisman"])
@pytest.mark.parametrize("name", MASS_ORBS)
def test_damage_orbs_hit_all_enemies_with_accuracy(kind, name):
    effect = EFFECTS[f"{kind} of {name}"]
    assert effect["scope"] == "party" and effect["accuracy"] == 80.0
    assert "scope" not in EFFECTS[f"{kind} of Thunder"]


def test_mass_orb_damages_every_enemy_through_armor():
    env = battle()
    assert env._apply_hero_item_effect(item_name="Orb of Inferno", target_pos=2)[0]
    # 75 damage, +0 jitter, 40 armor -> 45 on each living enemy.
    assert [env._unit_by_position(p)["health"] for p in (1, 2, 3)] == [955, 955, 955]


def test_defending_target_takes_half():
    env = battle()
    env._unit_by_position(3)["defense"] = 1
    env._apply_hero_item_effect(item_name="Orb of Inferno", target_pos=1)
    assert [env._unit_by_position(p)["health"] for p in (1, 2, 3)] == [955, 955, 978]


def test_orb_strikes_can_miss():
    env = battle(rng_value=0.99)  # (99 + 99) / 2 is not below 80.
    env._apply_hero_item_effect(item_name="Orb of Inferno", target_pos=1)
    assert [env._unit_by_position(p)["health"] for p in (1, 2, 3)] == [1000, 1000, 1000]


def test_thunder_still_hits_only_the_chosen_enemy():
    env = battle()
    env._apply_hero_item_effect(item_name="Orb of Thunder", target_pos=2)
    assert [env._unit_by_position(p)["health"] for p in (1, 2, 3)] == [1000, 940, 1000]


def test_vampire_orb_drains_every_enemy_and_heals_half():
    env = battle(enemies=(1, 2))
    for pos in (1, 2):
        env._unit_by_position(pos)["armor"] = 0
    hero = env._unit_by_position(8)
    hero["health"] = 10
    env._apply_hero_item_effect(item_name="Orb of Vampire", target_pos=1)
    assert [env._unit_by_position(p)["health"] for p in (1, 2)] == [925, 925]
    # drainAttackHeal = 50: half of the 150 total damage.
    assert hero["health"] == 10 + 75


def test_elder_vampire_orb_shares_the_half_beyond_the_hero():
    env = battle(enemies=(1, 2))
    for pos in (1, 2):
        env._unit_by_position(pos)["armor"] = 0
    hero = env._unit_by_position(8)
    hero["health"] = hero["max_health"] - 25
    ally = env._unit_by_position(7)
    ally["health"] = ally["max_health"] - 100
    env._apply_hero_item_effect(item_name="Orb of Elder Vampires", target_pos=1)
    assert hero["health"] == hero["max_health"]
    assert ally["health"] == ally["max_health"] - 100 + (75 - 25)


@pytest.mark.parametrize("name,multiplier", [("Orb of Strength", 1.5), ("Talisman of Vigor", 1.25)])
def test_boost_orbs_buff_every_ally(name, multiplier):
    env = battle()
    allies = [env._unit_by_position(p) for p in (7, 8, 11)]
    base = [unit["damage"] for unit in allies]
    assert env._apply_hero_item_effect(item_name=name, target_pos=7)[0]
    assert [unit["damage"] for unit in allies] == [round(d * multiplier) for d in base]


@pytest.mark.parametrize("name", ["Orb of Life", "Talisman of Life"])
def test_life_orb_revives_with_half_health(name):
    env = battle()
    corpse = env._unit_by_position(7)
    corpse["health"] = corpse["hp"] = 0
    corpse["initiative"] = 0
    assert env._apply_hero_item_effect(item_name=name, target_pos=7)[0]
    assert corpse["health"] == round(corpse["max_health"] / 2)
