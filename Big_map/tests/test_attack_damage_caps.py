"""Native attack caps at production boundaries, without erasing raw stat layers.

Expected 300/400 values and eligible hero IDs are independent of the cap module.
The catalogue matrix exercises real attack dispatch for all 252 fighters / 52
runtime types. Tracing wraps (never replaces) production damage calculation;
focused cases also assert HP, public BLUE steps and scheduled RED activations.
No training, replays, or scripted campaign bot are started.
"""
from copy import deepcopy
import random

import pytest

from attack_damage_limits import (
    attack_damage_cap, effective_attack_damage, effective_primary_amount,
)
from battle_env import BattleEnv, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from permanent_unit_stats import add_permanent_effect
from unit_dynamic_stats import DYNAMIC_STAT_PROFILES


CATALOG = {row["кто"]: row for row in DATA}
NATIVE_NAMES = tuple(CATALOG)
HEAVY_HERO_IDS = {
    "g000uu0019", "g000uu0020", "g000uu0044", "g000uu0045", "g000uu0047",
    "g000uu0070", "g000uu0071", "g000uu0096", "g000uu8009", "g000uu8011",
}
HERO_IDS = HEAVY_HERO_IDS | {
    "g000uu0021", "g000uu0022", "g000uu0046", "g000uu0072", "g000uu0073",
    "g000uu0097", "g000uu0098", "g000uu0099", "g000uu8010", "g000uu8012",
}
TEMPLATES = {name: map_unit_to_battle(row, "blue", 7)
             for name, row in CATALOG.items()}
HERO_NAMES = tuple(name for name, u in TEMPLATES.items() if u["unit_id"] in HERO_IDS)
HEAL_TYPES = {"Cliric", "Profit", "Patriach", "Deva roshi", "Sundancer", "Sylfid"}
POINT_SUPPORT_TYPES = {
    "Cliric", "Patriach", "Deva roshi", "Travnitsa", "Novice", "Dwarfdruid",
    "Arhidruid", "Alchemist",
}
SUMMON_TYPES = {"Summoner", "Occultmaster", "Lyf", "Laclaan"}
AOE_DAMAGE_TYPES = {
    "Mage", "Dead dragon", "Wolf Lord", "Gumtic", "Drulliaan", "Uter Demon",
    "Tiamat", "Teurg", "Hermit", "Vampire", "Highvampire", "Shamanka",
}
NON_OFFENSIVE_TYPES = {
    "Cliric", "Profit", "Patriach", "Deva roshi", "Sundancer", "Sylfid",
    "Ghost", "Witch", "Shadow", "Incub", "Succub", "Summoner", "Occultmaster",
    "Laclaan", "Lyf", "Baroness", "Travnitsa", "Novice", "Alchemist",
    "Dwarfdruid", "Arhidruid", "Doppelganger",
}


class FixedRandom(random.Random):
    def __init__(self, jitter=0):
        super().__init__(42)
        self.jitter = jitter

    def random(self):
        return 0.0

    def randint(self, low, high):
        return self.jitter if (low, high) == (0, 5) else low

    def choice(self, values):
        return values[0]

    def shuffle(self, values):
        pass


class TracedBattle(BattleEnv):
    def __init__(self, jitter=0):
        super().__init__(log_enabled=False)
        self.rng = FixedRandom(jitter)
        self.damage_events = []

    def _apply_damage_with_armor(self, attacker, base_dmg, victim):
        result = super()._apply_damage_with_armor(attacker, base_dmg, victim)
        self.damage_events.append((attacker["position"], victim["position"], result))
        return result


def native(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


def make_battle(name, side="blue", *, amount=450, enemies=1, jitter=0):
    battle = TracedBattle(jitter)
    own = 7 if side == "blue" else 1
    opposing = 1 if side == "blue" else 7
    enemy_side = "red" if side == "blue" else "blue"
    attacker = native(name, side, own)
    attacker.update(damage=amount, original_damage=amount,
                    accuracy=100, accuracy_secondary=0)
    ally = native("Рыцарь", side, own + 1)
    ally.update(health=100, max_health=10000, damage=100, original_damage=100,
                immunity=[], resistance=[], armor=0)
    victims = [native("Рыцарь", enemy_side, opposing + i) for i in range(enemies)]
    for victim in victims:
        victim.update(health=10000, max_health=10000, armor=0,
                      immunity=[], resistance=[], resilience_used_types=[])
    roster = [attacker, ally, *victims]
    occupied = {unit["position"] for unit in roster}
    battle.combined = roster + [
        placeholder_unit("red" if p < 7 else "blue", p)
        for p in range(1, 13) if p not in occupied
    ]
    return battle, attacker, ally, victims


def exercise_native_route(name, side, amount):
    """Use native dispatch; copy is a separate scheduler branch in production."""
    battle, attacker, ally, victims = make_battle(name, side, amount=amount)
    unit_type = attacker["unit_type"]
    expected_cap = 400 if attacker["unit_id"] in HEAVY_HERO_IDS else 300
    assert attack_damage_cap(attacker) == expected_cap
    expected_visible = amount if unit_type in NON_OFFENSIVE_TYPES else min(amount, expected_cap)
    assert effective_primary_amount(attacker) == expected_visible
    before = deepcopy(attacker)
    target_pos = victims[0]["position"]
    if unit_type == "Doppelganger":
        assert battle._apply_doppelganger_copy(attacker, victims[0])
        assert attacker["unit_id"] == before["unit_id"]
        return
    if unit_type in POINT_SUPPORT_TYPES:
        target_pos += 1  # Opposite cell maps to the wounded ally.
    elif unit_type == "Summoner":
        target_pos += 2  # Opposite cell is a free friendly slot.
    battle._attack(attacker, target_pos)
    if unit_type in NON_OFFENSIVE_TYPES:
        assert all(event[2] == 0 for event in battle.damage_events)
        if unit_type in HEAL_TYPES:
            assert ally["health"] == 100 + amount
    else:
        assert len(battle.damage_events) == 1
        assert battle.damage_events[0][2] == min(amount, expected_cap)
        critical = .05 * min(amount, expected_cap) if unit_type == "Centaur Savage" else 0
        assert victims[0]["health"] == 10000 - min(amount, expected_cap) - critical
    assert attacker["damage"] == before["damage"]
    assert attacker["original_damage"] == before["original_damage"]


def test_catalogue_coverage_is_all_252_units_52_types_and_20_heroes():
    assert len(NATIVE_NAMES) == 252
    assert len({u["unit_type"] for u in TEMPLATES.values()}) == 52
    assert len(HERO_NAMES) == 20
    assert {TEMPLATES[name]["unit_id"] for name in HERO_NAMES} == HERO_IDS
    assert sum(DYNAMIC_STAT_PROFILES[u["unit_id"]][3] == 1
               for u in TEMPLATES.values()) == 214


@pytest.mark.parametrize("name", NATIVE_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
def test_every_native_unit_uses_its_production_attack_or_support_route(name, side):
    exercise_native_route(name, side, 450)


@pytest.mark.parametrize("name", HERO_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("amount", [350, 450])
def test_all_twenty_hero_identities_at_both_cap_boundaries(name, side, amount):
    exercise_native_route(name, side, amount)


@pytest.mark.parametrize("side", ["blue", "red"])
def test_native_lord_170_doubled_by_archdruid_hits_for_300(side):
    battle, attacker, ally, victims = make_battle("Владыка", side, amount=170)
    druid = native("Архидруид", side, ally["position"])
    assert battle._apply_travnitsa_buff(druid, attacker)
    assert attacker["damage"] == 340
    assert battle._attack(attacker, victims[0]["position"])[0]
    assert victims[0]["health"] == 9700
    assert (attacker["damage"], attacker["original_damage"]) == (340, 170)
    battle._reset_powerup(attacker)
    assert attacker["damage"] == 170


def grow_to(battle, unit, level):
    while unit["Level"] < level:
        before = unit["Level"]
        unit["exp_current"] = unit["exp_required"] - 1
        battle._apply_exp_award_to_unit(unit, 1)
        assert unit["Level"] == before + 1


@pytest.mark.parametrize("side", ["blue", "red"])
def test_level44_archlich_keeps_raw_305_but_deals_300_to_each_of_six_targets(side):
    battle, attacker, _, victims = make_battle("Архилич", side, amount=90, enemies=6)
    grow_to(battle, attacker, 44)
    assert attacker["damage"] == 305
    assert battle._attack(attacker, None)[0]
    assert [victim["health"] for victim in victims] == [9700] * 6
    assert sum(event[2] for event in battle.damage_events) == 1800
    assert attacker["damage"] == 305
    grow_to(battle, attacker, 45)
    assert attacker["damage"] == 310
    assert effective_attack_damage(attacker) == 300


@pytest.mark.parametrize("name", [name for name, u in TEMPLATES.items()
                                  if u["unit_type"] in HEAL_TYPES])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_every_native_healer_keeps_500_healing_and_display_amount(name, side):
    exercise_native_route(name, side, 500)


@pytest.mark.parametrize("name,dot", [("Владыка", "burn"), ("Смерть", "poison"),
                                     ("Сын Имира", "uran")])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_native_secondary_dot_500_bypasses_primary_attack_cap(name, dot, side):
    battle, attacker, _, victims = make_battle(name, side)
    attacker.update(damage_secondary=500, accuracy_secondary=100)
    victim = victims[0]
    assert battle._attack(attacker, victim["position"])[0]
    assert victim["health"] == 9700
    assert victim[dot + "_damage_per_tick"] == 500
    victim["round_effects_done"] = 0
    assert battle._apply_start_of_turn_effects(victim)
    assert victim["health"] == 9200
    assert attacker["damage_secondary"] == 500


@pytest.mark.parametrize("name,cap", [("Рыцарь", 300), ("Герцог", 400)])
@pytest.mark.parametrize("armor,defense,multiplier,expected_factor", [
    (0, 0, 1, 1), (90, 0, 1, .1), (150, 0, 1, .1),
    (0, 1, 1, .5), (90, 1, 1, .05), (0, 0, .5, .5),
])
@pytest.mark.parametrize("jitter", [0, 5])
def test_cap_precedes_jitter_armor_defense_and_incoming_multiplier(
    name, cap, armor, defense, multiplier, expected_factor, jitter,
):
    battle, attacker, _, victims = make_battle(name, amount=999, jitter=jitter)
    victim = victims[0]
    victim.update(armor=armor, defense=defense, incoming_damage_multiplier=multiplier)
    assert battle._attack(attacker, victim["position"])[0]
    expected = round((cap + jitter) * expected_factor)
    assert 10000 - victim["health"] == expected
    assert attacker["damage"] == 999


@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("defence", ["immunity", "resistance"])
def test_caps_preserve_immunity_and_once_only_ward(side, defence):
    battle, attacker, _, victims = make_battle("Герцог", side)
    victim = victims[0]
    victim[defence] = [attacker["attack_type_primary"]]
    assert battle._attack(attacker, victim["position"])[0]
    assert victim["health"] == 10000
    assert battle._attack(attacker, victim["position"])[0]
    assert victim["health"] == (10000 if defence == "immunity" else 9600)


@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("armor,expected", [(0, 315), (90, 45)])
def test_centaur_critical_is_five_percent_of_capped_characteristic(side, armor, expected):
    battle, attacker, _, victims = make_battle("Кентавр дикарь", side, amount=350)
    attacker["accuracy_secondary"] = 100
    victims[0]["armor"] = armor
    assert battle._attack(attacker, victims[0]["position"])[0]
    assert 10000 - victims[0]["health"] == expected
    assert attacker["damage"] == 350


@pytest.mark.parametrize("name", ["Мастер клинка", "Бандит"])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_double_strike_scheduler_applies_the_cap_to_each_strike(name, side):
    battle, source, _, victims = make_battle(name, side, amount=450)
    # Add a BLUE sentinel so the real scheduler stops before a new round.
    stopper = native("Крестьянин", "blue", 12)
    stopper.update(health=100000, max_health=100000)
    battle.combined = [u for u in battle.combined if u["position"] != 12] + [stopper]
    battle._init_with_custom_teams(
        [u for u in battle.combined if u["team"] == "red"],
        [u for u in battle.combined if u["team"] == "blue"],
    )
    source = battle._unit_by_position(source["position"])
    for unit in battle.combined:
        unit.update(initiative=0, round_effects_done=1)
    source["initiative"] = 100
    battle._unit_by_position(12)["initiative"] = 90
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    battle.damage_events.clear()
    assert battle._advance_until_blue_turn()
    if side == "blue":
        assert battle.current_blue_attacker_pos == source["position"]
        for _ in range(2):
            observation, _, done, truncated, _ = battle.step(
                TARGET_POSITIONS.index(victims[0]["position"]))
            assert not done and not truncated
            assert observation.shape == battle.observation_space.shape
    assert battle.current_blue_attacker_pos == 12
    hits = [event[2] for event in battle.damage_events if event[0] == source["position"]]
    assert hits == [300, 300]
    assert sum(hits) == 600


@pytest.mark.parametrize("side", ["blue", "red"])
def test_debuff_uses_uncapped_450_so_half_damage_is_225_and_cleanse_restores(side):
    battle, attacker, _, victims = make_battle("Рыцарь", side, amount=225)
    assert battle._apply_hero_item_damage_buff(
        target_unit=attacker, effect={"damage_multiplier": 2}) == 225
    assert attacker["damage"] == 450
    assert effective_attack_damage(attacker) == 300
    assert battle._apply_hero_item_damage_debuff(
        source_unit=None, target_unit=attacker,
        effect={"damage_multiplier": .5}) == 225
    assert attacker["damage"] == 225
    assert battle._attack(attacker, victims[0]["position"])[0]
    assert victims[0]["health"] == 9775
    assert battle._cleanse_negative_effects(attacker)
    assert attacker["damage"] == 450
    assert effective_attack_damage(attacker) == 300
    battle._reset_powerup(attacker)
    assert attacker["damage"] == 225


def test_permanent_growth_above_cap_survives_repeated_form_restore():
    battle, attacker, _, victims = make_battle("Герцог", amount=450)
    add_permanent_effect(attacker, {"kind": "damage", "multiplier": 1.1})
    assert attacker["damage"] == 495
    original_id = attacker["unit_id"]
    for _ in range(3):
        assert battle._apply_hero_item_lycanthropy(
            source_unit=None, target_unit=attacker,
            effect={"transform_unit_name": "Оборотень"}) == 1
        attacker["damage"] = 500
        assert attack_damage_cap(attacker) == 300
        assert attacker["unit_id"] == original_id
        assert battle._attack(attacker, victims[0]["position"])[0]
        assert battle._restore_transformed_unit(attacker)
        assert attacker["unit_id"] == original_id
        assert attacker["damage"] == 495
        assert effective_attack_damage(attacker) == 400
    grow_to(battle, attacker, 2)
    assert attacker["damage"] == 506  # (450 + native hero increment 10) x1.1.
    assert effective_attack_damage(attacker) == 400


@pytest.mark.parametrize("target_name,expected", [("Герцог", 400), ("Рыцарь", 300)])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_doppelganger_cap_follows_copied_form_without_changing_permanent_identity(
    target_name, expected, side,
):
    battle, copier, _, victims = make_battle("Двойник", side, amount=0)
    target = native(target_name, victims[0]["team"], victims[0]["position"])
    target.update(damage=450, original_damage=450, health=10000, max_health=10000,
                  armor=0, immunity=[], resistance=[])
    battle.combined = [target if u is victims[0] else u for u in battle.combined]
    original_id = copier["unit_id"]
    for _ in range(3):
        assert battle._apply_doppelganger_copy(copier, target)
        assert copier["unit_id"] == original_id
        assert attack_damage_cap(copier) == expected
        assert copier["damage"] == 450
        copier["accuracy"] = 100
        before = target["health"]
        assert battle._attack(copier, target["position"])[0]
        assert before - target["health"] == expected
        battle._restore_default_doppelgangers((copier,))
        assert copier["unit_id"] == original_id
        assert attack_damage_cap(copier) == 300
        assert not copier["doppel_copied"]


@pytest.mark.parametrize("overrides", [
    {"name": "Герцог", "hero": True},
    {"name": "Лесной лорд", "hero": True},
    {"unit_id": "g000uu5008", "name": "Лесной лорд", "hero": True},
    {"unit_id": "unknown", "hero": True, "unit_type": "Warrior"},
])
def test_display_name_hero_flag_and_neutral_elf_lord_do_not_grant_heavy_strike(overrides):
    battle, attacker, _, victims = make_battle("Рыцарь", amount=450)
    attacker.update(overrides)
    assert battle._attack(attacker, victims[0]["position"])[0]
    assert victims[0]["health"] == 9700


def test_native_cap_does_not_limit_explicit_hero_item_damage():
    battle, attacker, _, victims = make_battle("Герцог", amount=450)
    assert battle._apply_hero_item_damage(
        source_unit=attacker, target_unit=victims[0],
        effect={"amount": 500, "attack_type": "Fire"}) == 500
    assert victims[0]["health"] == 9500
    assert not battle.damage_events


@pytest.mark.parametrize("name", [name for name, u in TEMPLATES.items()
                                  if u["unit_type"] in AOE_DAMAGE_TYPES])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_each_native_mass_attack_caps_each_target_instead_of_the_total(name, side):
    battle, attacker, _, victims = make_battle(name, side, amount=450, enemies=6)
    assert battle._attack(attacker, None)[0]
    assert [event[2] for event in battle.damage_events] == [300] * 6
    assert [victim["health"] for victim in victims] == [9700] * 6
    assert attacker["damage"] == 450


@pytest.mark.parametrize("name", ["Вампир", "Высший вампир"])
@pytest.mark.parametrize("side", ["blue", "red"])
def test_vampire_healing_can_exceed_cap_when_summing_six_capped_hits(name, side):
    battle, attacker, _, victims = make_battle(name, side, amount=450, enemies=6)
    attacker.update(health=1, max_health=10000)
    assert battle._attack(attacker, None)[0]
    assert [victim["health"] for victim in victims] == [9700] * 6
    assert attacker["health"] == 901  # Half the actual 1800 HP removed.


@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("outer_form", ["witch", "lycanthropy"])
def test_nested_copy_and_transform_restore_each_form_cap_in_order(side, outer_form):
    battle, copier, _, victims = make_battle("Двойник", side, amount=0)
    target = native("Герцог", victims[0]["team"], victims[0]["position"])
    target.update(damage=450, original_damage=450, health=10000, max_health=10000)
    assert battle._apply_doppelganger_copy(copier, target)
    assert attack_damage_cap(copier) == 400
    original_id = copier["unit_id"]
    if outer_form == "witch":
        battle._apply_witch_effect({}, copier)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=None, target_unit=copier,
            effect={"transform_unit_name": "Оборотень"}) == 1
    copier["damage"] = 500
    assert attack_damage_cap(copier) == 300
    assert battle._attack(copier, victims[0]["position"])[0]
    assert victims[0]["health"] == 9700
    assert battle._restore_transformed_unit(copier)
    assert copier["damage"] == 450
    assert attack_damage_cap(copier) == 400
    assert battle._attack(copier, victims[0]["position"])[0]
    assert victims[0]["health"] == 9300
    battle._restore_default_doppelgangers((copier,))
    assert copier["unit_id"] == original_id
    assert attack_damage_cap(copier) == 300


@pytest.mark.parametrize("name,cap", [("Рыцарь", 300), ("Герцог", 400)])
@pytest.mark.parametrize("amount", [-5, 0, 1, 299, 300, 301, 399, 400, 401, 1000])
def test_production_attack_boundary_values_do_not_change_raw_characteristics(name, cap, amount):
    battle, attacker, _, victims = make_battle(name, amount=amount)
    assert battle._attack(attacker, victims[0]["position"])[0]
    assert 10000 - victims[0]["health"] == max(0, min(amount, cap))
    assert attacker["damage"] == amount


def test_local5_party_strength_uses_effective_attack_and_uncapped_healing():
    from types import SimpleNamespace
    from local_observation import LocalObservation

    roster = [native(name, "blue", 7 + i)
              for i, name in enumerate(["Рыцарь", "Герцог", "Архангел"])]
    for unit in roster:
        unit["damage"] = 500
    observation = object.__new__(LocalObservation)
    observation.env = SimpleNamespace(_get_blue_state=lambda: roster)
    encoded = observation.party_strength().reshape(6, 4)
    assert encoded[:3, 1] == pytest.approx([300 / 400, 400 / 500, 500 / 600])
    assert [unit["damage"] for unit in roster] == [500, 500, 500]


def test_map_spell_damage_500_remains_independent_of_native_attack_caps():
    from campaign_env import CampaignEnv

    env = object.__new__(CampaignEnv)
    victims = [native("Рыцарь", "red", i) for i in (1, 2)]
    for unit in victims:
        unit.update(health=1000, max_health=1000, armor=90,
                    immunity=[], resistance=[])
    env.enemy_team_states = {123: victims}
    assert env._apply_damage_to_enemy_stack(123, 500, "Fire") == (1000, 2, 0, 0)
    assert [unit["health"] for unit in victims] == [500, 500]


@pytest.mark.parametrize("side", ["blue", "red"])
def test_campaign_save_reload_keeps_over_cap_growth_and_discards_temporary_form(side):
    import json
    from campaign_env import CampaignEnv

    env = CampaignEnv(map_name="scroll_train", observation_version="local5",
                      scripted_capital_bot_enabled=False,
                      use_boss_starting_roster=False, log_enabled=False)
    try:
        env.reset(seed=42)
        position = 7 if side == "blue" else 1
        original = native("Герцог", side, position)
        original.update(damage=450, original_damage=450)
        add_permanent_effect(original, {"kind": "damage", "multiplier": 1.1})
        enemy_id = 7717
        if side == "blue":
            env.blue_team_state = env._build_battle_team_with_placeholders(side, [original])
        else:
            env.enemy_team_states[enemy_id] = [deepcopy(original)]
        for iteration in range(3):
            battle = TracedBattle()
            roster = env._build_battle_team_with_placeholders(side, [deepcopy(original)])
            battle._init_with_custom_teams(
                roster if side == "red" else [], roster if side == "blue" else [])
            runtime = battle._unit_by_position(position)
            if iteration == 0:
                grow_to(battle, runtime, 2)
            assert runtime["damage"] == 506
            assert battle._apply_hero_item_damage_buff(
                target_unit=runtime, effect={"damage_multiplier": 2}) > 0
            assert runtime["damage"] == 1012
            assert effective_attack_damage(runtime) == 400
            battle._apply_witch_effect({}, runtime)
            assert attack_damage_cap(runtime) == 300
            env.battle_env = battle
            if side == "blue":
                env._save_blue_state()
                saved = next(u for u in env.blue_team_state if u["position"] == position)
            else:
                env._save_enemy_state_from_battle(enemy_id)
                saved = next(u for u in env.enemy_team_states[enemy_id]
                             if u["position"] == position)
            assert saved["damage"] == 506
            assert saved["Level"] == 2
            assert saved["unit_id"] == "g000uu0070"
            assert attack_damage_cap(saved) == 400
            assert "_attack_form_unit_id" not in saved
            original = json.loads(json.dumps(saved))
    finally:
        env.close()
