"""A copied summoner uses its attack form, retaining the copier's identity."""
import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS, WAIT_ACTION_INDEX
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


CATALOG = {entry["кто"]: entry for entry in DATA}
SUMMONS = [("Элементалист", "Элементаль Воздуха"),
           ("Оккультист", "Зомби"), ("Мудрец", "Малый энт")]


def unit(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


@pytest.mark.parametrize("name,spawn_name", SUMMONS)
def test_round_zero_copy_has_working_masked_summon_action(name, spawn_name):
    battle = BattleEnv(log_enabled=True)
    battle.seed(7)
    red = [unit("Скваер", "red", 4)]
    blue = [unit("Двойник", "blue", 7), unit(name, "blue", 10)]
    red += [placeholder_unit("red", p) for p in (1, 2, 3, 5, 6)]
    blue += [placeholder_unit("blue", p) for p in (8, 9, 11, 12)]
    battle._init_with_custom_teams(red, blue)
    copier = battle._unit_by_position(7)
    own_id = copier["unit_id"]
    assert battle.round_no == 0 and battle.current_blue_attacker_pos == 7
    assert battle.compute_action_mask()[TARGET_POSITIONS.index(10)]
    battle.step(TARGET_POSITIONS.index(10))
    for _ in range(12):
        if battle.current_blue_attacker_pos == 7:
            break
        battle.step(WAIT_ACTION_INDEX if battle.compute_action_mask()[WAIT_ACTION_INDEX]
                    else DEFEND_ACTION_INDEX)
    assert battle.current_blue_attacker_pos == 7
    assert copier["unit_type"] == "Summoner"
    assert copier["name"] == "Двойник" and copier["unit_id"] == own_id
    action = next(index for index in range(12) if battle.compute_action_mask()[index])
    before = sum(bool(u.get("Summoned")) for u in battle.combined)
    _, reward, terminated, truncated, _ = battle.step(action)
    summoned = [u for u in battle.combined if u.get("Summoned") == 7]
    assert len(summoned) == before + 1 == 1
    assert summoned[0]["name"] == spawn_name
    assert reward == battle.reward_step
    assert not terminated and not truncated
    assert copier["name"] == "Двойник" and copier["unit_id"] == own_id


@pytest.mark.parametrize("name,spawn_name", SUMMONS)
@pytest.mark.parametrize("team", ["blue", "red"])
def test_summon_dispatch_uses_canonical_form_id_even_with_a_different_name(name, spawn_name, team):
    battle = BattleEnv(log_enabled=False)
    position, target_pos = (7, 2) if team == "blue" else (1, 11)
    copier = unit("Двойник", team, position)
    target = unit(name, team, position + 1)
    # The alternate native Occultist ID must work too.
    if name == "Оккультист":
        target["unit_id"] = "g000uu6113"
    target["name"] = "Custom display name"
    battle.combined = [copier]
    assert battle._apply_doppelganger_copy(copier, target)
    assert battle._attack(copier, target_pos) == (True, "ok")
    spawned = next(u for u in battle.combined if u.get("Summoned"))
    assert spawned["name"] == spawn_name
    assert spawned["team"] == team
    battle._apply_hero_item_damage(source_unit=None, target_unit=copier,
                                 effect={"amount": 100000})
    assert spawned["health"] == 0
