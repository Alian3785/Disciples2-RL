"""B11: Hag's Ring identifies the five capital guardians by stable profile ID.

Exercise the real proc and attack paths; only deterministic RNG is substituted.
Faction (capital), combat type, display name and copied stats are not identity.
"""

from copy import deepcopy
import random

import pytest

from battle_env import BattleEnv, HAGS_RING_ARTIFACT_ITEM_NAME
from data_dicts_compact_lines import DATA, map_unit_to_battle

UNITS = {entry["кто"]: entry for entry in DATA}
GUARDIANS = {
    "Мизраэль": "g000uu3001",
    "Видар": "g000uu3002",
    "Ашкаэль": "g000uu3003",
    "Ашган": "g000uu3004",
    "Иллюмиэлль": "g000uu8040",
}
RING = HAGS_RING_ARTIFACT_ITEM_NAME


class FixedRandom(random.Random):
    def __init__(self, value=0.0):
        super().__init__(42)
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def make_battle(name, attacker_team="blue"):
    battle = BattleEnv(log_enabled=False)
    battle.rng = FixedRandom()
    victim_team = "red" if attacker_team == "blue" else "blue"
    attacker = map_unit_to_battle(UNITS["Герцог"], attacker_team, 7 if attacker_team == "blue" else 1)
    attacker.update(accuracy=100, campaign_active_artifacts=[RING])
    victim = map_unit_to_battle(UNITS[name], victim_team, 1 if victim_team == "red" else 7)
    battle.combined = [attacker, victim]
    return battle, attacker, victim


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name,profile_id", GUARDIANS.items())
@pytest.mark.parametrize("identity", ["canonical", "renamed", "mojibake", "uppercase_id", "game_unit_id", "legacy_name", "legacy_blank_id"])
def test_guardian_proc_changes_no_stats_or_status(team, name, profile_id, identity):
    battle, attacker, victim = make_battle(name, team)
    assert victim["unit_id"] == profile_id
    assert (victim["damage"], victim["armor"], victim["unit_type"]) == (250, 50, "Mage")
    if identity == "renamed":
        victim["name"] = "Custom guardian name"
    elif identity == "mojibake":
        victim["name"] = name.encode("cp1251").decode("latin1")
    elif identity == "uppercase_id":
        victim.update(name="renamed", unit_id=" " + profile_id.upper() + " ")
    elif identity == "game_unit_id":
        victim.pop("unit_id")
        victim.update(name="renamed", game_unit_id=profile_id)
    elif identity == "legacy_name":
        victim.pop("unit_id")
    elif identity == "legacy_blank_id":
        victim["unit_id"] = ""
    # A rejected transform must not consume a ward or overwrite existing effects.
    victim.update(resistance=["Mind"], powerup=1, original_damage=250, hermited=1)
    victim["basestats"] = [{"initiative_base": 91, "damage": 250}]
    before = deepcopy(victim)
    assert not battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim == before
    assert battle.rng.calls == 2  # The unchanged 70% proc still rolls once.


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", GUARDIANS)
def test_equipped_ring_attack_keeps_guardian_stats(team, name):
    battle, attacker, victim = make_battle(name, team)
    before = deepcopy(victim)
    assert battle._attack(attacker, victim["position"]) == (True, "ok")
    assert victim["health"] < before["health"]  # The ordinary hit still lands.
    for key in ("damage", "armor", "unit_type", "initiative_base", "transformed", "basestats", "unit_id"):
        assert victim[key] == before[key]
    assert battle.rng.calls == 5  # Hit + damage bonus + artifact proc.


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", ["Рыцарь", "Одержимый", "Гном", "Воин", "Кентавр", "Демон", "Маг"])
def test_ordinary_targets_still_transform_after_equipped_ring_hit(team, name):
    battle, attacker, victim = make_battle(name, team)
    assert battle._attack(attacker, victim["position"])[0]
    assert victim["transformed"] == 1
    assert victim["damage"] == (30 if victim["big"] else 20)
    assert victim["armor"] == 0
    assert victim["unit_type"] == "Warrior"


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", GUARDIANS)
def test_display_name_does_not_turn_an_ordinary_unit_into_guardian(team, name):
    battle, attacker, victim = make_battle("Рыцарь", team)
    victim["name"] = name
    assert battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim["transformed"] == 1


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", GUARDIANS)
def test_original_guardian_profile_remains_protected_after_temporary_form(team, name):
    battle, attacker, victim = make_battle(name, team)
    # Existing item transformation retains identity even while changing name/type.
    assert battle._apply_hero_item_lycanthropy(
        source_unit=attacker, target_unit=victim,
        effect={"damage_type": "Mind", "transform_unit_name": "Оборотень"},
    )
    assert victim["unit_id"] == GUARDIANS[name]
    before = deepcopy(victim)
    assert not battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim == before


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", GUARDIANS)
def test_doppelganger_copying_guardian_stats_is_not_guardian(team, name):
    battle, attacker, guardian = make_battle(name, team)
    victim = map_unit_to_battle(UNITS["Двойник"], guardian["team"], guardian["position"] + 1)
    battle.combined.append(victim)
    original_id = victim["unit_id"]
    assert battle._apply_doppelganger_copy(victim, guardian)
    assert victim["unit_id"] == original_id
    assert (victim["damage"], victim["armor"]) == (250, 50)
    assert battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert (victim["damage"], victim["armor"]) == (20, 0)


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("victim_name", ["Мизраэль", "Рыцарь"])
@pytest.mark.parametrize("block", ["failed_proc", "dead", "mind_immunity"])
def test_existing_proc_life_and_immunity_gates_unchanged(team, victim_name, block):
    battle, attacker, victim = make_battle(victim_name, team)
    if block == "failed_proc":
        battle.rng.value = 0.99
    elif block == "dead":
        victim["health"] = 0
    else:
        victim["immunity"] = ["mInD"]
        victim["resistance"] = ["Mind"]
    before = deepcopy(victim)
    assert not battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim == before
    assert battle.rng.calls == (0 if block == "dead" else 2)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_ordinary_mind_ward_blocks_once_then_transform_applies(team):
    battle, attacker, victim = make_battle("Рыцарь", team)
    victim["resistance"] = ["Mind"]
    assert not battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim["resilience_used_types"] == ["Mind"]
    assert victim["transformed"] == 0
    assert battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim["transformed"] == 1
    assert battle.rng.calls == 4


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("eligibility", ["no_ring", "not_hero", "duplicate_ring"])
def test_artifact_ownership_and_deduplication_unchanged(team, eligibility):
    battle, attacker, victim = make_battle("Рыцарь", team)
    if eligibility == "no_ring":
        attacker["campaign_active_artifacts"] = []
    elif eligibility == "not_hero":
        attacker["hero"] = False
    else:
        attacker["campaign_active_artifacts"] = [RING, RING]
    battle._apply_hero_artifact_post_hit_statuses(attacker, victim)
    assert victim["transformed"] == (1 if eligibility == "duplicate_ring" else 0)
    assert battle.rng.calls == (2 if eligibility == "duplicate_ring" else 0)


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 79, 101])
def test_rng_sequence_matches_unmodified_proc_contract(seed):
    battle, attacker, victim = make_battle("Мизраэль")
    battle.rng = random.Random(seed)
    expected = random.Random(seed)
    expected.random()
    expected.random()
    before = deepcopy(victim)
    assert not battle._apply_hero_artifact_transform(attacker, victim, RING)
    assert victim == before
    assert battle.rng.getstate() == expected.getstate()
