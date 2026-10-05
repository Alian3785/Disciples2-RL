"""Support effects agree through BLUE step and the scheduled RED controller.

This is an execution-parity regression, not an original-game rules audit. RED
keeps its existing target policy (notably: point healers need wounded targets).
All support casts below use actual scheduling, never a stubbed advance method.
"""
from copy import deepcopy
import random

import pytest

from battle_env import (
    BattleEnv, DEFEND_ACTION_INDEX, MASS_HEAL_TYPES, SUPPORT_TYPES,
    TARGET_POSITIONS,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

CATALOG = {row["кто"]: row for row in DATA}
SUPPORT_CATALOG = {
    "Cliric": ("Служка", "Жрец", "Священник", "Архангел", "Медиум", "Оракул"),
    "Profit": ("Клирик", "Аббатиса", "Прорицательница", "Нейтральный эльфийский оракул"),
    "Patriach": ("Патриарх",),
    "Travnitsa": ("Травница",),
    "Novice": ("Посвященная",),
    "Alchemist": ("Алхимик",),
    "Dwarfdruid": ("Друид",),
    "Arhidruid": ("Архидруид",),
    "Deva roshi": ("Дева рощи",),
    "Sundancer": ("Солнечная танцовщица",),
    "Sylfid": ("Сильфида",),
}
ALL_SUPPORT_NAMES = tuple(name for names in SUPPORT_CATALOG.values() for name in names)
CURE_NAMES = ("Аббатиса", "Прорицательница", "Друид", "Архидруид")
BUFF_MULTIPLIERS = {"Травница": 1.25, "Посвященная": 1.5, "Друид": 1.75, "Архидруид": 2}
WARD_ELEMENTS = {
    "Дева рощи": ("Fire", "Air", "Water", "Earth"),
    "Солнечная танцовщица": ("Fire",),
    "Сильфида": ("Air",),
}
STATUS_FIELDS = (
    "poison_turns_left", "poison_damage_per_tick", "burn_turns_left",
    "burn_damage_per_tick", "uran_turns_left", "uran_damage_per_tick",
    "paralyzed", "long_paralyzed", "feared", "running_away", "hermited",
    "teamated", "shattered_armor",
)
STATE_FIELDS = STATUS_FIELDS + (
    "health", "max_health", "damage", "original_damage", "armor", "initiative",
    "initiative_base", "powerup", "bonusturn", "transformed", "immunity",
    "resistance", "resilience_used_types", "Firedefence", "Airdefence",
    "Waterdefence", "Earthdefence",
)


class FixedRng(random.Random):
    def randint(self, low, high):
        return low

    def random(self):
        return 0.99  # No incidental natural recovery.

    def choice(self, sequence):
        return sequence[0]

    def shuffle(self, values):
        pass


class RecordedBattle(BattleEnv):
    def __init__(self):
        super().__init__(log_enabled=False)
        self.attack_events = []

    def _attack(self, attacker, target_pos):
        self.attack_events.append((attacker["team"], attacker["position"], target_pos))
        return super()._attack(attacker, target_pos)


def native(name, team, position):
    unit = map_unit_to_battle(CATALOG[name], team, position)
    unit.update(health=1000, max_health=1000, accuracy=100,
                accuracy_secondary=100, immunity=[], resistance=[],
                initiative_base=20, initiative=20)
    return unit


def case(name, team, target_slot=1):
    """Initialize normally, then place a mid-round state before the next cast.

    A BLUE sentinel stops the real scheduler after one support activation. No
    recipient can act before the assertion; restored spent actions stay visible.
    """
    own = 6 if team == "blue" else 0
    enemy_team = "red" if team == "blue" else "blue"
    enemy_pos = 1 if team == "blue" else 7
    unused = [slot for slot in range(1, 7) if slot not in (4, target_slot)]
    units = {
        "caster": native(name, team, own + 4),
        "target": native("Рыцарь", team, own + target_slot),
        "other": native("Крестьянин", team, own + unused[0]),
        "corpse": native("Крестьянин", team, own + unused[1]),
        "enemy": native("Рыцарь", enemy_team, enemy_pos),
    }
    occupied = {u["position"] for u in units.values()}
    stop_pos = next(pos for pos in range(12, 6, -1) if pos not in occupied)
    units["stopper"] = native("Крестьянин", "blue", stop_pos)
    units["stopper"]["initiative_base"] = 1000
    occupied.add(stop_pos)
    roster = list(units.values()) + [
        placeholder_unit("red" if p < 7 else "blue", p)
        for p in range(1, 13) if p not in occupied
    ]
    battle = RecordedBattle()
    battle.rng = FixedRng(0)
    battle._init_with_custom_teams(
        [u for u in roster if u["team"] == "red"],
        [u for u in roster if u["team"] == "blue"],
    )
    units = {key: battle._unit_by_position(u["position"]) for key, u in units.items()}
    for unit in battle.combined:
        unit.update(initiative=0, round_effects_done=1)
    units["caster"]["initiative"] = 100
    units["stopper"].update(initiative=90, damage=0, original_damage=0)
    units["other"].update(damage=0, original_damage=0)
    units["target"].update(health=100, damage=100, original_damage=100)
    units["corpse"].update(health=0, initiative=0)
    units["enemy"].update(health=100, damage=999)
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    battle.attack_events.clear()
    return battle, units


def cast(battle, units, recipient=None):
    caster = units["caster"]
    recipient = units["target"] if recipient is None else recipient
    if caster["team"] == "blue":
        assert battle._advance_until_blue_turn()
        assert battle.current_blue_attacker_pos == caster["position"]
        selector = recipient["position"] - 6
        observation, _, terminated, truncated, _ = battle.step(TARGET_POSITIONS.index(selector))
        assert observation.shape == battle.observation_space.shape
        assert not terminated and not truncated
    else:
        assert battle._advance_until_blue_turn()
    assert battle.round_no == 1
    assert battle.current_blue_attacker_pos == units["stopper"]["position"]


def state(unit):
    result = {key: deepcopy(unit.get(key, 0)) for key in STATE_FIELDS}
    wards = deepcopy(unit.get("_temporary_healer_wards", {}))
    for ward in wards.values():
        ward["sources"] = [(("own" if team == unit["team"] else "opponent"), (pos - 1) % 6 + 1)
                           for team, pos in ward["sources"]]
    result["wards"] = wards
    return result


def negative_state(unit, *, fear=True):
    unit.update(
        poison_turns_left=3, poison_damage_per_tick=7,
        burn_turns_left=3, burn_damage_per_tick=11,
        uran_turns_left=3, uran_damage_per_tick=13,
        paralyzed=1, long_paralyzed=1, feared=int(fear), running_away=int(fear),
        hermited=1, hermit_original_initiative_base=20, initiative_base=10,
        teamated=1, lower_damage_original_damage=100,
        lower_damage_original_unit_type=unit["unit_type"], damage=50,
        original_damage=100, armor=5, shattered_armor=20,
        shatter_original_armor=25, shatter_original_unit_type=unit["unit_type"],
    )


def assert_cleansed(unit):
    assert all(unit.get(key, 0) == 0 for key in STATUS_FIELDS)
    assert unit["initiative_base"] == 20
    assert unit["initiative"] == 0  # B06: no phantom activation before round end.
    assert unit["armor"] == 25
    for key in ("hermit_original_initiative_base", "lower_damage_original_damage",
                "lower_damage_original_unit_type", "shatter_original_armor",
                "shatter_original_unit_type"):
        assert key not in unit


def test_catalog_inventory_covers_every_support_name_and_type():
    actual = {}
    for row in DATA:
        if row["тип"] in SUPPORT_TYPES:
            actual.setdefault(row["тип"], []).append(row["кто"])
    assert {kind: tuple(names) for kind, names in actual.items()} == SUPPORT_CATALOG


@pytest.mark.parametrize("name", ALL_SUPPORT_NAMES)
def test_all_catalog_support_effects_match_blue_step_and_scheduled_red(name):
    results = []
    for team in ("blue", "red"):
        battle, units = case(name, team)
        target = units["target"]
        # Keep Patriarch's live-heal branch distinct from its revival tests.
        if name == "Патриарх":
            units["corpse"]["health"] = 1000
        target.update(poison_turns_left=3, poison_damage_per_tick=7)
        before_enemy = state(units["enemy"])
        cast(battle, units)
        assert state(units["enemy"]) == before_enemy
        assert target["poison_turns_left"] == (0 if name in CURE_NAMES else 3)
        if name in BUFF_MULTIPLIERS:
            assert target["damage"] == round(100 * BUFF_MULTIPLIERS[name])
            assert target["powerup"] == 1
            assert target["health"] == 100
        elif name == "Алхимик":
            assert (target["initiative"], target["bonusturn"], target["health"]) == (20, 1, 100)
        else:
            assert target["health"] == 100 + units["caster"]["damage"]
        assert target["resistance"] == list(WARD_ELEMENTS.get(name, ()))
        results.append(state(target))
    assert results[0] == results[1]


@pytest.mark.parametrize("name", CURE_NAMES)
@pytest.mark.parametrize("full_health", (False, True))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_entire_cleanse_bundle_and_original_heal_or_buff(name, full_health, team):
    battle, units = case(name, team)
    target = units["target"]
    negative_state(target)
    target["health"] = 1000 if full_health else 100
    # Each real source cache must be released by the public support action.
    for cache in (battle._spider_applied_poison, battle._dregazul_applied_poison,
                  battle._lord_applied_burn, battle._ismir_applied_uran):
        cache[id(units["enemy"])] = (units["enemy"], target)
    cast(battle, units)
    assert_cleansed(target)
    assert target not in battle._candidates()
    assert all(not cache for cache in (battle._spider_applied_poison,
               battle._dregazul_applied_poison, battle._lord_applied_burn,
               battle._ismir_applied_uran))
    if name in BUFF_MULTIPLIERS:
        assert target["damage"] == round(100 * BUFF_MULTIPLIERS[name])
        assert target["health"] == (1000 if full_health else 100)
    else:
        assert target["damage"] == 100
        assert target["health"] == (1000 if full_health else 100 + units["caster"]["damage"])


@pytest.mark.parametrize("name", ("Аббатиса", "Прорицательница"))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_mass_cure_covers_living_allies_but_not_caster_dead_or_enemy(name, team):
    battle, units = case(name, team)
    for key in ("target", "other", "corpse", "enemy"):
        negative_state(units[key])
    units["other"]["health"] = 1000
    units["caster"].update(health=100, poison_turns_left=3, poison_damage_per_tick=7)
    excluded = {key: state(units[key]) for key in ("corpse", "enemy")}
    cast(battle, units)
    assert_cleansed(units["target"])
    assert_cleansed(units["other"])
    for key, before in excluded.items():
        assert state(units[key]) == before
    assert (units["caster"]["health"], units["caster"]["poison_turns_left"]) == (100, 3)


@pytest.mark.parametrize("name", ("Друид", "Архидруид"))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_point_cure_is_single_target_and_can_cleanse_without_buff_damage(name, team):
    battle, units = case(name, team)
    for key in ("target", "other", "corpse", "enemy"):
        negative_state(units[key])
    units["target"].update(damage=0, original_damage=0, teamated=0)
    units["target"].pop("lower_damage_original_damage")
    units["target"].pop("lower_damage_original_unit_type")
    units["other"].update(damage=0, original_damage=0)
    excluded = {key: state(units[key]) for key in ("other", "corpse", "enemy")}
    cast(battle, units)
    assert_cleansed(units["target"])
    assert units["target"]["damage"] == 0
    assert units["target"].get("powerup", 0) == 0
    for key, before in excluded.items():
        assert state(units[key]) == before


@pytest.mark.parametrize("name,alias", (("Аббатиса", "Matriarch"), ("Прорицательница", "Prophetess")))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_imported_english_mass_cure_names_keep_the_same_bundle(name, alias, team):
    battle, units = case(name, team)
    units["caster"]["name"] = alias
    negative_state(units["target"])
    cast(battle, units)
    assert_cleansed(units["target"])


@pytest.mark.parametrize("name", ("Служка", "Дева рощи", "Друид", "Архидруид", "Алхимик", "Патриарх"))
@pytest.mark.parametrize("slot", (1, 2, 3, 5, 6))
def test_scheduled_red_target_mapping_preserves_formation_slot(name, slot):
    battle, units = case(name, "red", slot)
    if name == "Патриарх":
        units["corpse"]["health"] = 1000
    cast(battle, units)
    assert units["other"]["health"] == 1000
    assert units["other"]["damage"] == 0
    assert units["other"].get("bonusturn", 0) == 0
    assert units["enemy"]["health"] == 100
    target = units["target"]
    assert target["health"] > 100 or target["damage"] > 100 or target["bonusturn"] == 1


@pytest.mark.parametrize("name", ("Служка", "Дева рощи", "Патриарх"))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_point_healers_can_heal_self(name, team):
    battle, units = case(name, team)
    for key in ("target", "other", "corpse"):
        units[key]["health"] = 1000
    units["caster"]["health"] = 100
    cast(battle, units, units["caster"])
    assert units["caster"]["health"] == 100 + units["caster"]["damage"]


@pytest.mark.parametrize("name", ALL_SUPPORT_NAMES)
def test_scheduled_red_never_helps_dead_units_except_patriarch(name):
    battle, units = case(name, "red")
    for key in ("target", "other", "corpse"):
        units[key]["health"] = 0
    dead_before = {key: state(units[key]) for key in ("target", "other", "corpse")}
    enemy_before = state(units["enemy"])
    cast(battle, units)
    if name == "Патриарх":
        assert units["target"]["health"] == 500
        assert units["target"]["initiative"] == 0
        dead_before.pop("target")
    for key, before in dead_before.items():
        assert state(units[key]) == before
    assert state(units["enemy"]) == enemy_before
    if name not in SUPPORT_CATALOG["Patriach"] and units["caster"]["unit_type"] not in MASS_HEAL_TYPES:
        assert not battle.attack_events


@pytest.mark.parametrize("team", ("blue", "red"))
def test_patriarch_revives_cleanses_and_does_not_restore_spent_activation(team):
    battle, units = case("Патриарх", team)
    target = units["target"]
    negative_state(target)
    target["health"] = 0
    # Avoid a tie with an unrelated corpse in the native revival selector.
    units["corpse"]["health"] = 1000
    cast(battle, units)
    assert target["health"] == 500
    assert_cleansed(target)
    assert battle._patriach_already_revived(target)
    assert target not in battle._candidates()


@pytest.mark.parametrize("name", tuple(WARD_ELEMENTS))
@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("native_used", (False, True))
def test_public_healer_wards_block_once_and_expire_with_native_consumption(name, team, native_used):
    battle, units = case(name, team)
    target = units["target"]
    elements = WARD_ELEMENTS[name]
    target["resistance"] = [elements[0], "Mind"]
    target["resilience_used_types"] = [elements[0]] if native_used else []
    cast(battle, units)
    for element in elements:
        assert target["_temporary_healer_wards"][element]["sources"] == [(team, units["caster"]["position"])]
        assert battle._resilience_blocks({"attack_type_primary": element}, target)
        assert not battle._resilience_blocks({"attack_type_primary": element}, target)
    # Expiry uses another real scheduled activation, not an explicit ward clear.
    # Remove valid wounded recipients, so RED does not immediately recast.
    for unit in battle.combined:
        if battle._alive(unit):
            unit["health"] = unit["max_health"]
        unit["initiative"] = 0
    caster = units["caster"]
    # A changed form must still expire grants using the original caster's slot.
    caster["unit_type"] = "Cliric"
    caster["initiative"] = 100
    units["stopper"]["initiative"] = 90
    battle.current_blue_attacker_pos = None
    assert battle._advance_until_blue_turn()
    if team == "blue":
        assert battle.current_blue_attacker_pos == caster["position"]
        battle.step(DEFEND_ACTION_INDEX)
    assert target["resistance"] == [elements[0], "Mind"]
    assert not target.get("_temporary_healer_wards")
    assert not battle._resilience_blocks({"attack_type_primary": elements[0]}, target)
    assert battle._resilience_blocks({"attack_type_primary": "Mind"}, target)
    assert target["immunity"] == []


@pytest.mark.parametrize("name", ("Солнечная танцовщица", "Сильфида"))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_mass_healers_grant_wards_to_full_hp_allies_without_cleansing(name, team):
    battle, units = case(name, team)
    for key in ("target", "other"):
        units[key].update(health=1000, poison_turns_left=3, poison_damage_per_tick=7)
    cast(battle, units)
    for key in ("target", "other"):
        assert units[key]["health"] == 1000
        assert units[key]["poison_turns_left"] == 3
        assert units[key]["resistance"] == list(WARD_ELEMENTS[name])
    assert not units["caster"].get("_temporary_healer_wards")
    assert not units["corpse"].get("_temporary_healer_wards")
    assert not units["enemy"].get("_temporary_healer_wards")


def test_full_hp_point_ward_is_manual_ability_but_red_keeps_wounded_target_policy():
    for team in ("blue", "red"):
        battle, units = case("Дева рощи", team)
        for key in ("target", "other"):
            units[key]["health"] = 1000
        cast(battle, units)
        if team == "blue":
            assert units["target"]["resistance"] == list(WARD_ELEMENTS["Дева рощи"])
        else:
            assert units["target"]["resistance"] == []
            assert units["caster"]["defense"] == 1
            assert not battle.attack_events


@pytest.mark.parametrize("name", CURE_NAMES)
@pytest.mark.parametrize("team", ("blue", "red"))
def test_cleanse_releases_only_its_target_dot_binding_and_allows_reapplication(name, team):
    battle, units = case(name, team)
    source, other_source = units["enemy"], units["other"]
    for unit in (source, other_source):
        unit.update(attack_type_secondary="Poison", accuracy_secondary=100, damage_secondary=7)
    cache = battle._spider_applied_poison
    battle._apply_cached_poison(source, units["target"], cache)
    battle._apply_cached_poison(other_source, units["enemy"], cache)
    assert cache[id(source)] == (source, units["target"])
    assert cache[id(other_source)] == (other_source, units["enemy"])
    battle._apply_cached_poison(source, units["other"], cache)
    assert units["other"]["poison_turns_left"] == 0
    cast(battle, units)
    assert id(source) not in cache
    assert cache[id(other_source)] == (other_source, units["enemy"])
    battle._apply_cached_poison(source, units["other"], cache)
    assert units["other"]["poison_turns_left"] > 0
    assert cache[id(source)] == (source, units["other"])


@pytest.mark.parametrize("name", ALL_SUPPORT_NAMES)
@pytest.mark.parametrize("team", ("blue", "red"))
def test_friendly_support_is_not_blocked_by_immunity_or_consumed_native_wards(name, team):
    battle, units = case(name, team)
    target = units["target"]
    target.update(immunity=["Life", "Mind"], resistance=["Life", "Death"],
                  resilience_used_types=["Death"])
    if name == "Патриарх":
        units["corpse"]["health"] = 1000
    cast(battle, units)
    assert target["health"] > 100 or target["damage"] > 100 or target["bonusturn"] == 1
    assert target["immunity"] == ["Life", "Mind"]
    assert target["resistance"] == ["Life", "Death", *WARD_ELEMENTS.get(name, ())]
    assert target["resilience_used_types"] == ["Death"]


@pytest.mark.parametrize("name", ("Травница", "Посвященная", "Друид", "Архидруид", "Алхимик"))
def test_manual_self_buff_never_grants_damage_or_extra_activation(name):
    battle, units = case(name, "blue")
    caster = units["caster"]
    caster.update(poison_turns_left=3, poison_damage_per_tick=7)
    before_damage = caster["damage"]
    cast(battle, units, caster)
    assert caster["damage"] == before_damage
    assert caster["initiative"] == 0
    assert caster["bonusturn"] == 0
    # Existing BLUE semantics allow the cure part even when self-buff is refused.
    assert caster["poison_turns_left"] == (0 if name in CURE_NAMES else 3)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("reason", ("self", "alchemist", "retreating"))
def test_alchemist_invalid_recipients_do_not_gain_an_extra_turn(team, reason):
    battle, units = case("Алхимик", team)
    for key in ("other", "corpse"):
        units[key]["health"] = 0
    target = units["target"]
    if reason == "self":
        target["health"] = 0
        target = units["caster"]
    elif reason == "alchemist":
        target["unit_type"] = "Alchemist"
    else:
        target.update(running_away=1, feared=0)
    cast(battle, units, target)
    assert target["bonusturn"] == 0
    assert target["initiative"] == 0


@pytest.mark.parametrize("name", CURE_NAMES)
@pytest.mark.parametrize("team", ("blue", "red"))
def test_cure_preserves_a_voluntary_retreat(name, team):
    battle, units = case(name, team)
    target = units["target"]
    target.update(poison_turns_left=3, poison_damage_per_tick=7, running_away=1, feared=0)
    cast(battle, units)
    assert target["poison_turns_left"] == 0
    assert target["running_away"] == 1


@pytest.mark.parametrize("name", CURE_NAMES)
def test_cure_restores_transformed_recipient_before_healing_or_buffing(name):
    results = []
    for team in ("blue", "red"):
        battle, units = case(name, team)
        target = units["target"]
        target.update(immunity=["Death"], resistance=["Mind"])
        battle._apply_witch_effect(units["enemy"], target)
        assert target["transformed"] == 1
        assert target["damage"] != 100
        cast(battle, units)
        assert target["transformed"] == 0
        assert target["basestats"] == {}
        assert target["immunity"] == ["Death"]
        assert target["resistance"] == ["Mind"]
        assert target["damage"] == round(100 * BUFF_MULTIPLIERS.get(name, 1))
        expected_heal = units["caster"]["damage"] if name not in BUFF_MULTIPLIERS else 0
        assert target["health"] == 100 + expected_heal
        results.append(state(target))
    assert results[0] == results[1]


@pytest.mark.parametrize("name", tuple(BUFF_MULTIPLIERS))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_damage_buff_expires_after_recipients_real_activation(name, team):
    battle, units = case(name, team)
    target = units["target"]
    cast(battle, units)
    assert target["damage"] == round(100 * BUFF_MULTIPLIERS[name])
    assert target["powerup"] == 1
    # Arrange the recipient's next legitimate activation without a round reset.
    target["initiative"] = 100
    units["enemy"]["health"] = 1000
    battle.current_blue_attacker_pos = None
    assert battle._advance_until_blue_turn()
    if team == "blue":
        assert battle.current_blue_attacker_pos == target["position"]
        battle.step(DEFEND_ACTION_INDEX)
    assert battle.current_blue_attacker_pos == units["stopper"]["position"]
    assert target["damage"] == 100
    assert target["powerup"] == 0
    assert target["initiative"] == 0
