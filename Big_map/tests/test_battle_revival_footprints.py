"""Revival must fit the restored fighter's whole formation footprint.

The orb reproduction uses native catalog stats and actual BattleEnv.step turns.
Narrow cases isolate the writers and both controllers; scheduling is never stubbed.
"""
from copy import deepcopy
import random

import pytest

from battle_env import (
    BattleEnv, DEFEND_ACTION_INDEX, FIRST_HERO_ITEM_ACTION_START,
    SECOND_HERO_ITEM_ACTION_START, TARGET_POSITIONS,
)
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

CATALOG = {row["кто"]: row for row in DATA}


class FixedRng(random.Random):
    def randint(self, low, high):
        return low

    def random(self):
        return 0.99  # Prevent incidental recovery of an intentionally held form.

    def choice(self, values):
        return values[0]

    def shuffle(self, values):
        pass


def native(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


def roster_with(*units):
    occupied = {unit["position"] for unit in units}
    return list(units) + [
        placeholder_unit("red" if pos < 7 else "blue", pos)
        for pos in range(1, 13) if pos not in occupied
    ]


def assert_legal_living_formation(battle, team):
    """Independent oracle: a paired cell may have only one living owner."""
    cells = []
    for unit in battle.combined:
        if unit["team"] != team or not battle._alive(unit):
            continue
        pos = unit["position"]
        cells.append(pos)
        if unit.get("big", False):
            cells.append(pos + 3 if (pos - 1) % 6 < 3 else pos - 3)
    assert len(cells) <= 6
    assert len(cells) == len(set(cells))
    return len(cells)


def advance_to(battle, position):
    for _ in range(24):
        if battle.current_blue_attacker_pos == position:
            return
        assert battle.compute_action_mask()[DEFEND_ACTION_INDEX]
        _, _, terminated, truncated, _ = battle.step(DEFEND_ACTION_INDEX)
        assert not terminated and not truncated
    raise AssertionError(f"No turn for position {position}")


def orb_case(front_name="Чёрт", front_alive=False):
    front = native(front_name, "blue", 7)
    if not front_alive:
        front["health"] = 0
    hero = native("Советник", "blue", 8)
    progression = BattleEnv(log_enabled=False)
    while hero["Level"] < 10:
        progression._apply_exp_award_to_unit(hero, int(hero["exp_required"]))
    roster = roster_with(front, hero, native("Одержимый", "blue", 9),
                         native("Сектант", "blue", 11), native("Сектант", "blue", 12),
                         native("Патриарх", "red", 4))
    battle = BattleEnv(log_enabled=False)
    battle.rng = FixedRng(0)
    battle.equipped_hero_items = ["Lich Orb", "Orb of Life"]
    battle.hero_item_effects = CampaignEnv._battle_item_effect_definitions()
    battle._init_with_custom_teams([u for u in roster if u["team"] == "red"],
                                   [u for u in roster if u["team"] == "blue"])
    assert battle.current_blue_attacker_pos == 8
    return battle


def summon_rear_lich(battle):
    action = FIRST_HERO_ITEM_ACTION_START + 3  # Blue position 10.
    assert battle.compute_action_mask()[action]
    info = battle.step(action)[4]
    assert info["battle_hero_item_applied"] and info["battle_hero_item_consumed"]
    lich = battle._unit_by_position(10)
    assert lich["name"] == "Лич" and battle._alive(lich)
    advance_to(battle, 8)
    return lich


@pytest.mark.parametrize("release", ("death", "disappearance"))
def test_public_lich_orb_blocks_large_revival_until_rear_cell_is_free(release):
    battle = orb_case()
    lich = summon_rear_lich(battle)
    corpse = battle._unit_by_position(7)
    revive = SECOND_HERO_ITEM_ACTION_START
    for _ in range(2):
        assert not battle.compute_action_mask()[revive]
        info = battle.step(revive)[4]  # A caller can submit a masked-out action.
        assert not info["battle_hero_item_applied"]
        assert not info["battle_hero_item_consumed"]
        assert not info["battle_hero_item_charge_spent"]
        assert battle.equipped_hero_items[1] == "Orb of Life"
        assert not battle.hero_item_slots_used_this_battle[1]
        assert corpse["health"] == 0 and battle._alive(lich)
        assert assert_legal_living_formation(battle, "blue") == 5
        advance_to(battle, 8)
    if release == "death":
        assert battle._apply_hero_item_damage(
            source_unit=None, target_unit=lich, effect={"amount": 100000}) > 0
        assert lich["health"] == 0
    else:
        battle._mark_unit_escaped(lich)
        assert lich["name"] == "пусто"
    assert battle.compute_action_mask()[revive]
    info = battle.step(revive)[4]
    assert info["battle_hero_item_applied"] and info["battle_hero_item_consumed"]
    assert corpse["health"] == 1
    assert battle.equipped_hero_items[1] is None
    assert assert_legal_living_formation(battle, "blue") == 6


def test_public_small_front_revival_remains_legal_beside_rear_lich():
    battle = orb_case("Одержимый")
    lich = summon_rear_lich(battle)
    action = SECOND_HERO_ITEM_ACTION_START
    assert battle.compute_action_mask()[action]
    info = battle.step(action)[4]
    assert info["battle_hero_item_applied"] and info["battle_hero_item_consumed"]
    assert battle._unit_by_position(7)["health"] == 1
    assert battle._alive(lich)
    assert assert_legal_living_formation(battle, "blue") == 6


def test_public_living_large_front_still_blocks_rear_summoning():
    battle = orb_case(front_alive=True)
    assert not battle.compute_action_mask()[FIRST_HERO_ITEM_ACTION_START + 3]
    assert assert_legal_living_formation(battle, "blue") == 6


def narrow_case(team, anchor, corpse_big=True, blocker_big=False):
    offset = 6 if team == "blue" else 0
    pos = offset + anchor
    paired = pos + 3 if anchor <= 3 else pos - 3
    corpse = native("Чёрт" if corpse_big else "Одержимый", team, pos)
    corpse.update(health=0, initiative=0)
    blocker = native("Чёрт" if blocker_big else "Одержимый", team, paired)
    healer = native("Патриарх", team, offset + 5)
    battle = BattleEnv(log_enabled=False)
    battle.rng = FixedRng(0)
    battle.combined = roster_with(corpse, blocker, healer)
    battle._patriach_revived_recipients = set()
    return battle, corpse, blocker, healer


@pytest.mark.parametrize("team", ("red", "blue"))
@pytest.mark.parametrize("anchor", (1, 4), ids=("front", "back"))
@pytest.mark.parametrize("writer", ("item", "patriarch"))
@pytest.mark.parametrize("corpse_big,blocker_big", ((True, False), (False, True)),
                         ids=("large_corpse", "large_blocker"))
def test_revival_writers_reject_overlapping_front_back_footprints(
        team, anchor, writer, corpse_big, blocker_big):
    battle, corpse, blocker, healer = narrow_case(team, anchor, corpse_big, blocker_big)
    before = deepcopy(corpse)
    for _ in range(2):
        if writer == "item":
            assert battle._apply_hero_item_revive(corpse, 1) == 0
        else:
            assert battle._apply_patriach_support(healer, corpse) == "invalid"
        assert corpse == before
        assert not battle._patriach_already_revived(corpse)
        assert_legal_living_formation(battle, team)
    blocker.update(health=0, initiative=0)
    if writer == "item":
        assert battle._apply_hero_item_revive(corpse, 1) == 1
        assert corpse["health"] == 1
    else:
        assert battle._apply_patriach_support(healer, corpse) == "revive_success"
        assert corpse["health"] == round(corpse["max_health"] / 2)
        assert battle._patriach_already_revived(corpse)
    assert corpse["initiative"] == 0
    assert_legal_living_formation(battle, team)


def scheduled_patriarch_case(team, anchor):
    offset = 6 if team == "blue" else 0
    target = native("Чёрт", team, offset + anchor)
    target.update(health=0, initiative=0)
    blocker_pos = target["position"] + (3 if anchor <= 3 else -3)
    blocker = native("Одержимый", team, blocker_pos)
    # A summoned blocker is ineligible for Patriarch revival after it dies.
    blocker["Summoned"] = offset + 5
    caster = native("Патриарх", team, offset + 5)
    stopper = native("Крестьянин", "blue", 12)
    stopper.update(initiative_base=1000, damage=0, original_damage=0)
    units = {"target": target, "blocker": blocker, "caster": caster, "stopper": stopper}
    if team == "blue":
        units["enemy"] = native("Крестьянин", "red", 2)
    roster = roster_with(*units.values())
    battle = BattleEnv(log_enabled=False)
    battle.rng = FixedRng(0)
    battle._init_with_custom_teams([u for u in roster if u["team"] == "red"],
                                   [u for u in roster if u["team"] == "blue"])
    return battle, {key: battle._unit_by_position(u["position"]) for key, u in units.items()}


def schedule_patriarch(battle, units, *, expected_mask):
    """Advance exactly one real support action, then stop at a BLUE sentinel."""
    for unit in battle.combined:
        unit.update(initiative=0, round_effects_done=battle.round_no)
    caster, target = units["caster"], units["target"]
    caster["initiative"] = 100
    units["stopper"]["initiative"] = 90
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    assert battle._advance_until_blue_turn()
    if caster["team"] == "blue":
        assert battle.current_blue_attacker_pos == caster["position"]
        action = TARGET_POSITIONS.index(target["position"] - 6)
        assert bool(battle.compute_action_mask()[action]) is expected_mask
        _, _, terminated, truncated, _ = battle.step(action)
        assert not terminated and not truncated
    assert battle.round_no == 1
    assert battle.current_blue_attacker_pos == units["stopper"]["position"]


@pytest.mark.parametrize("team", ("red", "blue"))
@pytest.mark.parametrize("anchor", (1, 4), ids=("front", "back"))
def test_patriarch_controllers_reject_then_revive_without_spending_allowance(team, anchor):
    battle, units = scheduled_patriarch_case(team, anchor)
    target, blocker, caster = units["target"], units["blocker"], units["caster"]
    for _ in range(2):
        assert target not in battle._patriach_targetable_allies(caster)
        assert battle._patriach_auto_target(caster) != target["position"]
        schedule_patriarch(battle, units, expected_mask=False)
        assert target["health"] == 0
        assert not battle._patriach_already_revived(target)
        assert_legal_living_formation(battle, team)
    blocker.update(health=0, initiative=0)
    assert target in battle._patriach_targetable_allies(caster)
    assert battle._patriach_auto_target(caster) == target["position"]
    schedule_patriarch(battle, units, expected_mask=True)
    assert target["health"] == round(target["max_health"] / 2)
    assert battle._patriach_already_revived(target)
    assert target["initiative"] == 0
    assert_legal_living_formation(battle, team)
    target["health"] = 0
    schedule_patriarch(battle, units, expected_mask=False)
    assert target["health"] == 0  # Successful revival still consumes its allowance.


@pytest.mark.parametrize("team", ("red", "blue"))
@pytest.mark.parametrize("writer", ("item", "patriarch"))
def test_transformed_small_corpse_reserves_native_large_footprint_on_revival(team, writer):
    battle, corpse, blocker, healer = narrow_case(team, 1)
    corpse["health"] = corpse["max_health"]
    corpse["immunity"] = []
    corpse["resistance"] = []
    assert battle._apply_hero_item_lycanthropy(
        source_unit=healer, target_unit=corpse,
        effect={"transform_unit_name": "Скваер"}) == 1
    assert corpse["transformed"] and not corpse["big"]
    corpse.update(health=0, initiative=0)
    snapshot = deepcopy(corpse)
    effect = {"kind": "revive", "amount": 1}
    for _ in range(2):
        assert not battle._hero_item_effect_can_apply_to_target(
            effect=effect, effect_kind="revive", target_unit=corpse)
        if writer == "item":
            assert battle._apply_hero_item_revive(corpse, 1) == 0
        else:
            assert battle._apply_patriach_support(healer, corpse) == "invalid"
        assert corpse == snapshot  # Masks/writers must not eagerly cleanse the form.
        assert not battle._patriach_already_revived(corpse)
    blocker.update(health=0, initiative=0)
    if writer == "item":
        assert battle._apply_hero_item_revive(corpse, 1) > 0
    else:
        assert battle._apply_patriach_support(healer, corpse) == "revive_success"
    assert corpse["health"] > 0 and corpse["big"] and not corpse["transformed"]
    assert_legal_living_formation(battle, team)


@pytest.mark.parametrize("team", ("red", "blue"))
def test_linked_summon_owner_death_frees_large_corpse_revival(team):
    battle, corpse, _, healer = narrow_case(team, 1)
    offset = 6 if team == "blue" else 0
    owner = battle._unit_by_position(offset + 2)
    owner.clear()
    owner.update(native("Оккультист", team, offset + 2))
    rear = battle._unit_by_position(offset + 4)
    rear.clear()
    rear.update(placeholder_unit(team, offset + 4))
    assert battle._apply_hero_item_summon(
        source_unit=owner, target_team=team, target_pos=offset + 4,
        target_unit=rear, effect={"summon_unit_name": "Лич"}) == 1
    assert rear["Summoned"] == owner["position"]
    assert battle._apply_patriach_support(healer, corpse) == "invalid"
    assert battle._apply_hero_item_damage(
        source_unit=None, target_unit=owner, effect={"amount": 100000}) > 0
    assert owner["health"] == rear["health"] == 0
    assert battle._apply_patriach_support(healer, corpse) == "revive_success"
    assert_legal_living_formation(battle, team)


@pytest.mark.parametrize("team", ("red", "blue"))
@pytest.mark.parametrize("writer", ("item", "patriarch"))
def test_native_small_corpse_in_large_form_can_revive_beside_rear_fighter(team, writer):
    battle, corpse, blocker, healer = narrow_case(team, 1, corpse_big=False)
    corpse["health"] = corpse["max_health"]
    corpse["immunity"] = []
    corpse["resistance"] = []
    assert battle._apply_hero_item_lycanthropy(
        source_unit=healer, target_unit=corpse,
        effect={"transform_unit_name": "Чёрт"}) == 1
    assert corpse["transformed"] and corpse["big"]
    corpse.update(health=0, initiative=0)
    before = deepcopy(corpse)
    assert battle._hero_item_effect_can_apply_to_target(
        effect={"kind": "revive", "amount": 1}, effect_kind="revive", target_unit=corpse)
    assert corpse == before
    if writer == "item":
        assert battle._apply_hero_item_revive(corpse, 1) > 0
    else:
        assert battle._apply_patriach_support(healer, corpse) == "revive_success"
    assert corpse["health"] > 0 and not corpse["big"] and not corpse["transformed"]
    assert battle._alive(blocker)
    assert_legal_living_formation(battle, team)


@pytest.mark.parametrize("team", ("red", "blue"))
@pytest.mark.parametrize("blocked", (True, False), ids=("occupied_rear", "free_rear"))
def test_post_victory_patriarch_controllers_respect_revival_footprint(team, blocked):
    battle, units = scheduled_patriarch_case(team, 1)
    target, blocker = units["target"], units["blocker"]
    target["initiative"] = 0
    if not blocked:
        blocker.update(health=0, initiative=0)
    for unit in battle.combined:
        if unit["team"] != team:
            unit.update(health=0, initiative=0)
    # Native victory transition schedules BLUE input and executes RED automatically.
    battle._check_victory_after_hit()
    if team == "blue":
        assert battle._post_victory_team == "blue"
        assert battle.current_blue_attacker_pos == units["caster"]["position"]
        action = TARGET_POSITIONS.index(target["position"] - 6)
        assert bool(battle.compute_action_mask()[action]) is not blocked
        _, _, terminated, truncated, info = battle.step(action)
        assert terminated and not truncated
        assert info["post_victory_heal_completed"]
        assert info["post_victory_healed_hp"] == (
            0 if blocked else round(target["max_health"] / 2))
    assert battle.winner == team
    assert target["health"] == (0 if blocked else round(target["max_health"] / 2))
    assert battle._patriach_already_revived(target) is not blocked
    assert_legal_living_formation(battle, team)
