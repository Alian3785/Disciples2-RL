"""DoT ownership is a live effect, never a permanent formation-slot lock.

These checks deliberately exercise attacks and effect-removal entry points,
without asserting the representation of the implementation's ownership cache.
"""

from copy import deepcopy
import pickle
import random

import pytest

from battle_env import (
    BURN_TURNS,
    POISON_TURNS,
    URAN_TURNS,
    BattleEnv,
    _apply_dynamic_unit_levelup,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from unit_dynamic_stats import DYNAMIC_STAT_PROFILES


DATA_BY_NAME = {unit["кто"]: unit for unit in DATA}
CACHED_UNITS = [
    ("Владыка", "burn"),
    ("Сатир", "poison"),
    ("Сын Имира", "uran"),
    ("Танатос", "poison"),
    ("Головорез", "poison"),
    ("Мантикора", "poison"),
    ("Гигантский чёрный паук", "poison"),
    ("Ниддог", "poison"),
    ("Граф Фламель Кроули", "poison"),
    ("Дрега Зул", "poison"),
]
UNCACHED_UNITS = [
    ("Имперский ассасин", "poison"),
    ("Смерть", "poison"),
    ("Змей ужаса", "poison"),
    ("Стингер", "poison"),
    ("Смотритель", "burn"),
    ("Часовой", "uran"),
    ("Драллиаан", "uran"),
    ("Зверь Галлеана", "poison"),
    ("Гамтик кровавый", "burn"),
]
MAX_TURNS = {"poison": POISON_TURNS, "burn": BURN_TURNS, "uran": URAN_TURNS}
TICK_IMMUNITY = {"poison": "Poison", "burn": "Fire", "uran": "Water"}


class ControlledRandom(random.Random):
    """Exercise actual chance logic, with independently selectable duration."""

    def __init__(self, value=0.0, duration=6):
        super().__init__(17)
        self.value = value
        self.duration = duration
        self.duration_bounds = []

    def random(self):
        return self.value

    def randint(self, low, high):
        if low == 1:
            self.duration_bounds.append((low, high))
            assert low <= self.duration <= high
            return self.duration
        return high


def make_unit(name, team, position):
    unit = map_unit_to_battle(DATA_BY_NAME[name], team, position)
    unit.update(
        health=10000,
        max_health=10000,
        armor=0,
        immunity=[],
        resistance=[],
        accuracy=100,
        accuracy_secondary=100,
    )
    return unit


def setup(name, dot, side):
    battle = BattleEnv(log_enabled=False)
    battle.rng = ControlledRandom()
    own_position = 7 if side == "blue" else 1
    enemy_position = 1 if side == "blue" else 7
    enemy_team = "red" if side == "blue" else "blue"
    sources = [make_unit(name, side, own_position + offset) for offset in (0, 1)]
    targets = [
        make_unit("Рыцарь", enemy_team, enemy_position + offset)
        for offset in (0, 1, 2)
    ]
    battle.combined = sources + targets
    occupied = {unit["position"] for unit in battle.combined}
    battle.combined += [
        placeholder_unit("red" if position < 7 else "blue", position)
        for position in range(1, 13)
        if position not in occupied
    ]
    battle._patriach_revived_recipients = set()
    return battle, sources, targets, dot


@pytest.fixture(params=CACHED_UNITS, ids=[name for name, _ in CACHED_UNITS])
def cached_case(request, side):
    return setup(*request.param, side)


@pytest.fixture(params=["blue", "red"])
def side(request):
    return request.param


def turns(target, dot):
    return int(target.get(f"{dot}_turns_left", 0))


def damage(target, dot):
    return int(target.get(f"{dot}_damage_per_tick", 0))


def effect(target, dot):
    return turns(target, dot), damage(target, dot)


def attack(battle, source, target):
    before = target["health"]
    assert battle._attack(source, target["position"])[0]
    # A blocked DoT must not suppress the direct hit.
    assert target["health"] < before


def tick(battle, target):
    target["round_effects_done"] = 0
    return battle._apply_start_of_turn_effects(target)


def revive(battle, target, method):
    if method == "item":
        assert battle._apply_hero_item_revive(target, 5000) == 5000
    else:
        healer = make_unit(
            "Патриарх", target["team"], 12 if target["team"] == "blue" else 6
        )
        assert battle._apply_patriach_support(healer, target) == "revive_success"


@pytest.mark.parametrize("removal", ["expiry", "cleanse", "immunity", "death"])
def test_only_finished_effect_releases_its_source(cached_case, removal):
    battle, (first, second), (target, other, fresh), dot = cached_case
    attack(battle, first, target)
    attack(battle, second, other)
    original = effect(target, dot)
    assert original[0] > 0 and turns(other, dot) > 0

    if removal == "expiry":
        other[f"{dot}_turns_left"] = 1
        assert tick(battle, other)
    elif removal == "cleanse":
        assert battle._cleanse_negative_effects(other)
    elif removal == "immunity":
        other["immunity"] = [TICK_IMMUNITY[dot]]
        assert tick(battle, other)
    else:
        battle._subtract_health(other, other["health"])
        other["initiative"] = 0

    attack(battle, first, fresh)
    assert turns(fresh, dot) == 0, "another caster's removal unlocked this caster"
    assert effect(target, dot) == original
    attack(battle, second, fresh)
    assert turns(fresh, dot) > 0, "finished effect did not release its own caster"


@pytest.mark.parametrize("removal", ["expiry", "cleanse", "immunity"])
def test_unowned_dot_removal_does_not_unlock_native_source(cached_case, removal):
    battle, (source, _), (target, unrelated, fresh), dot = cached_case
    attack(battle, source, target)
    unrelated[f"{dot}_turns_left"] = 1
    unrelated[f"{dot}_damage_per_tick"] = 7
    if removal == "cleanse":
        assert battle._cleanse_negative_effects(unrelated)
    else:
        if removal == "immunity":
            unrelated["immunity"] = [TICK_IMMUNITY[dot]]
        assert tick(battle, unrelated)
    assert turns(unrelated, dot) == 0
    attack(battle, source, fresh)
    assert turns(fresh, dot) == 0
    assert turns(target, dot) > 0


@pytest.mark.parametrize("different_caster", [False, True])
def test_active_effect_is_not_refreshed_or_overwritten(cached_case, different_caster):
    battle, (source, other_source), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    assert tick(battle, target)
    original = effect(target, dot)
    attacker = other_source if different_caster else source
    attacker["damage_secondary"] = original[1] + 91
    battle.rng.duration = 2
    attack(battle, attacker, target)
    assert effect(target, dot) == original
    attack(battle, attacker, fresh)
    assert (turns(fresh, dot) > 0) is different_caster
    if different_caster:
        assert effect(fresh, dot) == (2, original[1] + 91)


@pytest.mark.parametrize("method", ["item", "patriarch"])
def test_source_death_and_revival_keep_live_effect_bound(cached_case, method):
    battle, (source, _), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    original = effect(target, dot)
    battle._subtract_health(source, source["health"])
    source["initiative"] = 0
    assert tick(battle, target)
    assert effect(target, dot) == (original[0] - 1, original[1])
    revive(battle, source, method)
    attack(battle, source, fresh)
    assert turns(fresh, dot) == 0
    assert turns(target, dot) > 0


@pytest.mark.parametrize("method", ["item", "patriarch"])
@pytest.mark.parametrize("death", ["direct", "early_tick", "last_tick"])
def test_target_death_and_revival_allow_reapplication(cached_case, method, death):
    battle, (source, _), (target, _, _), dot = cached_case
    attack(battle, source, target)
    if death == "direct":
        battle._subtract_health(target, target["health"])
        target["initiative"] = 0
    else:
        target["health"] = damage(target, dot)
        if death == "last_tick":
            target[f"{dot}_turns_left"] = 1
        assert not tick(battle, target)
    assert target["health"] == 0
    revive(battle, target, method)
    assert effect(target, dot) == (0, 0)
    attack(battle, source, target)
    assert effect(target, dot) == (6, source["damage_secondary"])


@pytest.mark.parametrize("replacement", ["new_dict", "reuse_dict"])
def test_new_source_in_same_slot_does_not_inherit_binding(cached_case, replacement):
    battle, (source, _), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    original = effect(target, dot)
    new_source = make_unit(source["name"], source["team"], source["position"])
    if replacement == "new_dict":
        battle.combined[battle.combined.index(source)] = new_source
    else:
        # Escape is an actual in-place slot replacement, used by the engine.
        battle._mark_unit_escaped(source)
        source.clear()
        source.update(new_source)
        new_source = source
    attack(battle, new_source, fresh)
    assert turns(fresh, dot) > 0
    assert effect(target, dot) == original


@pytest.mark.parametrize("replacement", ["new_dict", "escape_and_reuse"])
def test_new_target_in_same_slot_releases_previous_binding(cached_case, replacement):
    battle, (source, _), (target, _, _), dot = cached_case
    attack(battle, source, target)
    new_target = make_unit("Рыцарь", target["team"], target["position"])
    if replacement == "new_dict":
        battle.combined[battle.combined.index(target)] = new_target
    else:
        battle._mark_unit_escaped(target)
        target.clear()
        target.update(new_target)
        new_target = target
    attack(battle, source, new_target)
    assert turns(new_target, dot) > 0


@pytest.mark.parametrize("removal", ["detached", "escaped"])
def test_target_removal_releases_source(cached_case, removal):
    battle, (source, _), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    if removal == "detached":
        battle.combined.remove(target)
        # The detached dict deliberately retains its health and effect fields.
        assert turns(target, dot) > 0 and target["health"] > 0
    else:
        battle._mark_unit_escaped(target)
    attack(battle, source, fresh)
    assert turns(fresh, dot) > 0


@pytest.mark.parametrize("serializer", ["deepcopy", "pickle"])
@pytest.mark.parametrize("state", ["active", "target_dead"])
def test_battle_copy_preserves_live_ownership_and_reapplication(cached_case, serializer, state):
    battle, (source, _), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    if state == "target_dead":
        battle._subtract_health(target, target["health"])
        target["initiative"] = 0
    copied = deepcopy(battle) if serializer == "deepcopy" else pickle.loads(pickle.dumps(battle))
    copied_source = copied._unit_by_position(source["position"])
    copied_target = copied._unit_by_position(target["position"])
    copied_fresh = copied._unit_by_position(fresh["position"])
    assert copied_source is not source and copied_target is not target

    attack(copied, copied_source, copied_fresh)
    assert (turns(copied_fresh, dot) > 0) is (state == "target_dead")
    if state == "active":
        copied._subtract_health(copied_target, copied_target["health"])
        copied_target["initiative"] = 0
        attack(copied, copied_source, copied_fresh)
        assert turns(copied_fresh, dot) > 0
        assert battle._alive(target), "copy mutations leaked into original battle"
    assert turns(fresh, dot) == 0


@pytest.mark.parametrize("reset_path", ["default", "custom"])
def test_new_battle_resets_effects_and_allows_reapplication(cached_case, reset_path, monkeypatch):
    battle, (source, _), (target, _, _), dot = cached_case
    attack(battle, source, target)
    assert turns(target, dot) > 0
    red = [unit for unit in battle.combined if unit["team"] == "red"]
    blue = [unit for unit in battle.combined if unit["team"] == "blue"]
    if reset_path == "default":
        monkeypatch.setattr("battle_env.UNITS_RED", red)
        monkeypatch.setattr("battle_env.UNITS_BLUE", blue)
        battle._reset_state()
    else:
        # Isolate initialization itself from automatic enemy turns that follow.
        monkeypatch.setattr(battle, "_advance_until_blue_turn", lambda: True)
        battle._init_with_custom_teams(red, blue)
    reset_source = battle._unit_by_position(source["position"])
    reset_target = battle._unit_by_position(target["position"])
    assert reset_source is not source and reset_target is not target
    for unit in battle.combined:
        assert all(effect(unit, kind) == (0, 0) for kind in MAX_TURNS)
    attack(battle, reset_source, reset_target)
    assert effect(reset_target, dot) == (6, reset_source["damage_secondary"])


def test_lyf_resummon_reuses_dictionary_but_not_old_spider_binding(side):
    battle, (old_source, _), (target, fresh, _), dot = setup("Сатир", "poison", side)
    own_position = old_source["position"]
    old_source.clear()
    old_source.update(placeholder_unit(side, own_position))
    lyf = make_unit("Тёмный эльф Лиф", side, own_position + 2)
    # Reserve other rows so the real summon action deterministically creates
    # exactly one large Spider in the empty front/back pair.
    for replacement in (
        lyf,
        make_unit("Рыцарь", side, own_position + 4),
        make_unit("Рыцарь", side, own_position + 5),
    ):
        slot = battle._unit_by_position(replacement["position"])
        battle.combined[battle.combined.index(slot)] = replacement
    assert battle._attack(lyf, target["position"])[0]
    spider = battle._unit_by_position(own_position)
    assert spider is old_source and spider["unit_type"] == "Spider"
    assert spider["Summoned"] == lyf["position"]
    attack(battle, spider, target)
    original = effect(target, dot)
    assert original == (6, 35)

    battle._subtract_health(spider, spider["health"])
    spider["initiative"] = 0
    assert battle._attack(lyf, target["position"])[0]
    replacement_spider = battle._unit_by_position(own_position)
    assert replacement_spider is spider, "this regression requires actual in-place reuse"
    assert battle._alive(replacement_spider)
    attack(battle, replacement_spider, fresh)
    assert effect(fresh, dot) == (6, 35)
    assert effect(target, dot) == original


@pytest.mark.parametrize(
    "blocker",
    ["status_roll", "status_immunity", "status_ward", "primary_roll", "primary_ward"],
)
def test_failed_attempt_does_not_bind_source(cached_case, blocker):
    battle, (source, _), (target, fresh, _), dot = cached_case
    # Separate primary and secondary types to isolate secondary status gates.
    source["attack_type_primary"] = "Weapon"
    if blocker == "status_roll":
        source["accuracy_secondary"] = 0
    elif blocker == "status_immunity":
        target["immunity"] = [source["attack_type_secondary"]]
    elif blocker == "status_ward":
        target["resistance"] = [source["attack_type_secondary"]]
    elif blocker == "primary_roll":
        source["accuracy"] = 0
    else:
        target["resistance"] = ["Weapon"]
    assert battle._attack(source, target["position"])[0]
    assert turns(target, dot) == 0
    source.update(accuracy=100, accuracy_secondary=100)
    attack(battle, source, fresh)
    assert turns(fresh, dot) > 0


@pytest.mark.parametrize("duration", [1, 3, 6])
def test_native_chance_duration_and_tick_damage_are_unchanged(cached_case, duration):
    battle, (source, _), (target, _, _), dot = cached_case
    native = map_unit_to_battle(DATA_BY_NAME[source["name"]], source["team"], source["position"])
    native_chance = native["accuracy_secondary"]
    source["accuracy_secondary"] = native_chance
    battle.rng = ControlledRandom(native_chance / 100, duration)
    attack(battle, source, target)
    assert turns(target, dot) == 0, "the strict native chance boundary must still fail"
    battle.rng.value = (native_chance - 1) / 100
    attack(battle, source, target)
    assert effect(target, dot) == (duration, native["damage_secondary"])
    assert battle.rng.duration_bounds == [(1, MAX_TURNS[dot])]
    before = target["health"]
    assert tick(battle, target)
    assert target["health"] == before - native["damage_secondary"]
    assert turns(target, dot) == duration - 1
    # Waiting/repeating an activation in the same round must not double-tick.
    assert battle._apply_start_of_turn_effects(target)
    assert target["health"] == before - native["damage_secondary"]


@pytest.mark.parametrize("next_level", [10, 11])
def test_reapplication_uses_grown_secondary_damage(cached_case, next_level):
    battle, (source, _), (target, fresh, _), dot = cached_case
    attack(battle, source, target)
    before = source["damage_secondary"]
    source["Level"] = next_level - 1
    threshold, early, late, _, secondary, _ = DYNAMIC_STAT_PROFILES[source["unit_id"]]
    assert secondary == 1
    increment = (early if next_level <= threshold else late)[1]
    _apply_dynamic_unit_levelup(source, source["exp_required"])
    assert source["damage_secondary"] == before + increment
    assert damage(target, dot) == before, "an active DoT snapshots its application damage"
    assert battle._cleanse_negative_effects(target)
    attack(battle, source, fresh)
    assert damage(fresh, dot) == before + increment


@pytest.mark.parametrize("name", [name for name, dot in CACHED_UNITS if dot == "poison"])
@pytest.mark.parametrize("stronger", [False, True])
def test_artifact_poison_replacement_releases_only_replaced_native_effect(name, stronger, side):
    battle, (source, artifact_source), (target, fresh, _), dot = setup(name, "poison", side)
    attack(battle, source, target)
    initial_damage = damage(target, dot)
    new_damage = initial_damage + 17 if stronger else initial_damage
    assert battle._apply_hero_artifact_poison(
        artifact_source, target, "Thanatos Blade (Artifact)", new_damage
    ) is stronger
    assert damage(target, dot) == new_damage
    attack(battle, source, fresh)
    assert (turns(fresh, dot) > 0) is stronger
    assert damage(target, dot) == new_damage


@pytest.mark.parametrize("power_delta", [-1, 0, 17], ids=["weaker", "equal", "stronger"])
def test_item_dot_override_releases_only_replaced_caster(cached_case, power_delta):
    battle, (source, unrelated_source), (target, unrelated, fresh), dot = cached_case
    attack(battle, source, target)
    attack(battle, unrelated_source, unrelated)
    assert tick(battle, target)
    original = effect(target, dot)
    unrelated_original = effect(unrelated, dot)
    battle.rng.duration = 2
    replacement_damage = original[1] + power_delta
    result = battle._apply_hero_item_dot(
        source_unit=source,
        target_unit=target,
        effect={
            "dot_type": dot,
            "amount": replacement_damage,
            "damage_type": source["attack_type_secondary"],
        },
    )
    if power_delta > 0:
        assert result == replacement_damage
        assert effect(target, dot) == (2, replacement_damage)
    else:
        assert result == 0
        assert effect(target, dot) == original
    attack(battle, unrelated_source, fresh)
    assert turns(fresh, dot) == 0
    assert effect(unrelated, dot) == unrelated_original
    attack(battle, source, fresh)
    assert (turns(fresh, dot) > 0) is (power_delta > 0)


@pytest.mark.parametrize(
    "first_name,first_dot,second_name,second_dot",
    [
        ("Сатир", "poison", "Владыка", "burn"),
        ("Владыка", "burn", "Сын Имира", "uran"),
        ("Сын Имира", "uran", "Сатир", "poison"),
    ],
)
def test_item_dot_override_keeps_other_effect_on_same_target_bound(
    first_name, first_dot, second_name, second_dot, side,
):
    battle, (source, old_second), (target, fresh, _), _ = setup(first_name, first_dot, side)
    second = make_unit(second_name, side, old_second["position"])
    battle.combined[battle.combined.index(old_second)] = second
    attack(battle, source, target)
    attack(battle, second, target)
    other_effect = effect(target, second_dot)
    replacement_damage = damage(target, first_dot) + 13
    assert battle._apply_hero_item_dot(
        source_unit=source,
        target_unit=target,
        effect={"dot_type": first_dot, "amount": replacement_damage},
    ) == replacement_damage
    attack(battle, source, fresh)
    assert turns(fresh, first_dot) > 0
    attack(battle, second, fresh)
    assert turns(fresh, second_dot) == 0
    assert effect(target, second_dot) == other_effect


@pytest.mark.parametrize("first_name,second_name", [("Сатир", "Дрега Зул"), ("Дрега Зул", "Сатир")])
def test_shared_poison_slot_does_not_allow_cross_cache_overwrite(first_name, second_name, side):
    battle, (source, other), (target, fresh, _), dot = setup(first_name, "poison", side)
    replacement = make_unit(second_name, side, other["position"])
    battle.combined[battle.combined.index(other)] = replacement
    attack(battle, source, target)
    original = effect(target, dot)
    replacement["damage_secondary"] = original[1] + 50
    attack(battle, replacement, target)
    assert effect(target, dot) == original
    attack(battle, replacement, fresh)
    assert damage(fresh, dot) == original[1] + 50


@pytest.mark.parametrize("name,dot", UNCACHED_UNITS, ids=[name for name, _ in UNCACHED_UNITS])
def test_uncached_roster_retains_multiple_targets_and_no_refresh(name, dot, side):
    battle, (source, _), (target, fresh, extra), dot = setup(name, dot, side)
    # Initial AOE must only see the first target; later revival adds a new one.
    fresh.update(health=0, initiative=0)
    extra.update(health=0, initiative=0)
    attack(battle, source, target)
    assert turns(target, dot) == 6
    assert tick(battle, target)
    original = effect(target, dot)
    assert battle._apply_hero_item_revive(fresh, 5000) == 5000
    source["damage_secondary"] += 23
    attack(battle, source, fresh)
    assert turns(fresh, dot) == 6
    assert damage(fresh, dot) == source["damage_secondary"]
    assert effect(target, dot) == original
