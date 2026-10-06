"""One-action damage buffs remain local to a form and expire in saved forms.

All status applications, consumed activations, cleanses, and round recovery use
production ``step`` and the ordinary RED controller. Native unit stats are used
except initiative fixtures that make intermediate states observable and a
20-point Prophetess heal in RED form-first cases (so native buff targeting selects
the 30-damage Imp rather than the healer). The terminal cases additionally start the enemy at 1 HP to finish through one legal attack.
RNG controls success/recovery; no action, AI, or scheduling method is mocked.
"""

from copy import deepcopy
import random

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from permanent_unit_stats import ensure_stat_sources


CATALOG = {entry["кто"]: entry for entry in DATA}


class FixedRng(random.Random):
    recovery_roll = 0.6

    def randint(self, low, high):
        return low

    def random(self):
        return self.recovery_roll

    def choice(self, values):
        return values[0]

    def shuffle(self, values):
        pass


class RecordedBattle(BattleEnv):
    """Observe completed native actions without changing their implementation."""

    def __init__(self):
        super().__init__(log_enabled=False)
        self.actions = []

    def _attack(self, attacker, target_pos):
        result = super()._attack(attacker, target_pos)
        self.actions.append((attacker["position"], deepcopy(self.combined)))
        return result


def native(name, team, position, **changes):
    unit = map_unit_to_battle(CATALOG[name], team, position)
    unit.update(changes)
    return unit


def case(team="blue", *, timing="before", active_cleanse=False,
         cleanser=True, source=False, terminal=False, buffer="Травница", level_up=False):
    offset = 6 if team == "blue" else 0
    opposite = "red" if team == "blue" else "blue"
    units = {
        "target": native("Старейшина", team, offset + 1),
        "buffer": native(buffer, team, offset + 5),
        "witch": native("Ведьма", opposite, 4 if team == "blue" else 10,
                        initiative_base=65 if timing == "before" else 80),
        "gate": native("Белый маг" if team == "blue" else "Рыцарь",
                       "blue", 8, initiative_base=2),
    }
    if source:
        ensure_stat_sources(units["target"])
    if level_up:
        units["target"]["exp_current"] = units["target"]["exp_required"] - 1
    if terminal:
        units["witch"]["health"] = 1
    if cleanser:
        units["cleanser"] = native("Прорицательница", team, offset + 6,
                                   initiative_base=60 if active_cleanse else 10)
        if team == "red" and timing == "after":
            units["cleanser"]["damage"] = 20
    if team == "red":
        units["pause"] = native("Рыцарь", "blue", 7,
                                initiative_base=55 if active_cleanse else 15)
    occupied = {unit["position"] for unit in units.values()}
    roster = list(units.values()) + [
        placeholder_unit("red" if position < 7 else "blue", position)
        for position in range(1, 13) if position not in occupied
    ]
    battle = RecordedBattle()
    battle.rng = FixedRng(0)
    battle._init_with_custom_teams(
        [unit for unit in roster if unit["team"] == "red"],
        [unit for unit in roster if unit["team"] == "blue"],
    )
    units = {key: battle._unit_by_position(unit["position"])
             for key, unit in units.items()}
    initial_blue = deepcopy([unit for unit in battle.combined if unit["team"] == "blue"])
    return battle, units, initial_blue


def step(battle, action, actor=None):
    if actor is not None:
        assert battle.current_blue_attacker_pos == actor["position"]
    assert battle.compute_action_mask()[action]
    result = battle.step(action)
    assert not result[3]
    return result


def target_action(battle, units):
    # BLUE support selectors refer to the opposite RED cell; RED uses the same
    # production support executor with its normal automatic recipient selection.
    if units["target"]["team"] == "blue":
        step(battle, TARGET_POSITIONS.index(1), units["buffer"])
    else:
        step(battle, TARGET_POSITIONS.index(1), units["witch"])


def action_target_state(battle, actor, target):
    state = next(state for position, state in reversed(battle.actions)
                 if position == actor["position"])
    return next(unit for unit in state if unit["position"] == target["position"])


def expire_target(battle, units):
    if units["target"]["team"] == "blue":
        step(battle, DEFEND_ACTION_INDEX, units["target"])
    else:
        assert battle.current_blue_attacker_pos == units["pause"]["position"]
    assert units["target"]["powerup"] == 0
    assert units["target"]["initiative"] == 0


def cleanse_target(battle, units):
    if units["target"]["team"] == "blue":
        step(battle, TARGET_POSITIONS.index(1), units["cleanser"])
    else:
        step(battle, DEFEND_ACTION_INDEX, units["pause"])
    assert not units["target"]["transformed"]


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("buffer,multiplier", [
    ("Травница", 1.25), ("Посвященная", 1.5), ("Друид", 1.75), ("Архидруид", 2.0),
])
def test_expiry_keeps_temporary_form_native_damage(team, buffer, multiplier):
    battle, units, _ = case(team, buffer=buffer)
    target_action(battle, units)
    buffed = action_target_state(battle, units["buffer"], units["target"])
    assert (buffed["damage"], buffed["powerup"]) == (round(80 * multiplier), 1)
    expire_target(battle, units)
    assert units["target"]["transformed"] == 1
    assert units["target"]["damage"] == 30


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("source", [False, True])
def test_cleanse_cannot_restore_expired_original_form_buff(team, source):
    battle, units, _ = case(team, source=source)
    target_action(battle, units)
    expire_target(battle, units)
    cleanse_target(battle, units)
    target = units["target"]
    assert (target["damage"], target["original_damage"], target["powerup"]) == (80, 80, 0)
    assert not target.get("_battle_damage_factors")
    assert target["initiative"] == 0


@pytest.mark.parametrize("team", ["blue", "red"])
def test_natural_next_round_recovery_cannot_restore_expired_buff(team):
    battle, units, _ = case(team, cleanser=False)
    target_action(battle, units)
    expire_target(battle, units)
    # Keep repeat casts behind the next activation. The real round boundary
    # and start-of-turn recovery still run through step without replacements.
    units["buffer"]["initiative_base"] = 1
    units["witch"]["initiative_base"] = 1
    battle.rng.recovery_roll = 0.0
    if team == "red":
        step(battle, DEFEND_ACTION_INDEX, units["pause"])
    step(battle, DEFEND_ACTION_INDEX, units["gate"])
    assert battle.round_no == 2
    target = units["target"]
    assert (target["transformed"], target["damage"], target["powerup"]) == (0, 80, 0)
    assert not target.get("_battle_damage_factors")
    if team == "blue":
        assert battle.current_blue_attacker_pos == target["position"]
        assert target["initiative"] > 0
    else:
        assert target["initiative"] == 0


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_natural_recovery_before_action_preserves_only_original_form_buff(team, timing):
    battle, units, _ = case(team, timing=timing)
    battle.rng.recovery_roll = 0.0
    target_action(battle, units)
    target = units["target"]
    expected_damage = 100 if timing == "before" else 80
    expected_powerup = int(timing == "before")
    if team == "blue":
        observed = target
    else:
        # RED has already used its restored attack; observe it before the
        # production end-of-action reset, then check its final expired state.
        observed = action_target_state(battle, target, target)
    assert (observed["transformed"], observed["damage"], observed["powerup"]) == (
        0, expected_damage, expected_powerup,
    )
    if team == "blue":
        step(battle, DEFEND_ACTION_INDEX, target)
    assert (target["damage"], target["powerup"]) == (80, 0)


@pytest.mark.parametrize("team", ["blue", "red"])
def test_cleanse_preserves_unspent_original_buff_until_its_action(team):
    battle, units, _ = case(team, active_cleanse=True)
    target_action(battle, units)
    if team == "blue":
        step(battle, TARGET_POSITIONS.index(1), units["cleanser"])
    target = units["target"]
    assert (target["transformed"], target["damage"], target["powerup"]) == (0, 100, 1)
    if team == "blue":
        step(battle, DEFEND_ACTION_INDEX, target)
    else:
        step(battle, DEFEND_ACTION_INDEX, units["pause"])
    assert (target["damage"], target["powerup"]) == (80, 0)
    assert not target.get("_battle_damage_factors")


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("buffer,multiplier", [("Травница", 1.25), ("Посвященная", 1.5)])
def test_buff_applied_in_temporary_form_uses_that_forms_damage(team, buffer, multiplier):
    battle, units, _ = case(team, timing="after", buffer=buffer)
    target_action(battle, units)
    buffed = action_target_state(battle, units["buffer"], units["target"])
    assert buffed["transformed"] == 1
    assert (buffed["damage"], buffed["powerup"]) == (round(30 * multiplier), 1)
    expire_target(battle, units)
    assert units["target"]["damage"] == 30
    cleanse_target(battle, units)
    assert (units["target"]["damage"], units["target"]["powerup"]) == (80, 0)


@pytest.mark.parametrize("team", ["blue", "red"])
def test_still_active_temporary_form_buff_does_not_leak_into_original(team):
    battle, units, _ = case(team, timing="after", active_cleanse=True)
    target_action(battle, units)
    if team == "blue":
        step(battle, TARGET_POSITIONS.index(1), units["cleanser"])
    target = units["target"]
    assert (target["transformed"], target["damage"], target["powerup"]) == (0, 80, 0)
    assert not target.get("_battle_damage_factors")


@pytest.fixture
def campaign():
    env = CampaignEnv(map_name="trade_train", observation_version="local5",
                      scripted_capital_bot_enabled=False,
                      use_boss_starting_roster=False, log_enabled=False)
    env.reset(seed=42)
    yield env
    env.close()


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("cleanse", [False, True])
@pytest.mark.parametrize("level_up", [False, True])
def test_real_victory_and_campaign_save_cannot_persist_expired_buff(
        campaign, source, cleanse, level_up):
    battle, units, initial = case(source=source, terminal=True,
                                 cleanser=cleanse, level_up=level_up)
    expected = deepcopy(units["target"])
    if level_up:
        BattleEnv(log_enabled=False)._apply_exp_award_to_unit(expected, 1)
        assert expected["Level"] == 5
        assert expected["damage"] > 80
    target_action(battle, units)
    expire_target(battle, units)
    if cleanse:
        cleanse_target(battle, units)
    action = next(index for index in range(len(TARGET_POSITIONS))
                  if battle.compute_action_mask()[index])
    step(battle, action, units["gate"])
    # Complete the actual post-victory healer phase instead of setting winner.
    for _ in range(10):
        if battle.winner:
            break
        assert battle._post_victory_team == "blue"
        step(battle, TARGET_POSITIONS.index(1))
    assert battle.winner == "blue"
    live_damage = units["target"]["damage"]
    assert not units["target"]["transformed"]
    campaign.blue_team_state = initial
    campaign.battle_env = battle
    campaign._save_blue_state()
    saved = next(unit for unit in campaign.blue_team_state if unit["position"] == 7)
    # Assert both boundaries after actually reaching save: source records can
    # mask an already-corrupt live damage value during persistence rebuilding.
    assert (live_damage, saved["damage"]) == (expected["damage"], expected["damage"])
    assert saved["Level"] == expected["Level"]
    assert saved["powerup"] == 0
    assert not saved.get("_battle_damage_factors")
    assert not saved["transformed"]
