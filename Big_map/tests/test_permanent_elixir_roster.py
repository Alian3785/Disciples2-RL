"""All DATA profiles retain source-driven doses across ordinary stat growth.

These are deterministic unit-level arithmetic checks, not simulated campaign
playthroughs or permission to dynamically level an evolution-only unit.
"""
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from unit_dynamic_stats import apply_dynamic_stat_growth

POTIONS = tuple(d for d in CampaignEnv.POTION_ITEM_DEFINITIONS if d.get("duration") == "permanent")
STATS = ("damage", "damage_secondary", "accuracy", "accuracy_secondary", "initiative_base",
         "armor", "max_health", "health")
UNITS = [map_unit_to_battle(d, "blue", 8) for d in DATA]
UNITS = [u for u in UNITS if u.get("max_health", 0) > 0]


def apply_all(unit):
    # The unit-only helper has no dependency on map state or inventory. Normal
    # actions, consumption and eligibility are covered in test_permanent_elixir_layers.
    env = object.__new__(CampaignEnv)
    for _ in range(2):
        for potion in POTIONS:
            env._apply_permanent_potion_bonus_to_unit(unit, potion)


@pytest.mark.parametrize("unit", UNITS, ids=lambda u: u["name"])
@pytest.mark.parametrize("next_level", [10, 11])
def test_every_roster_profile_replays_doses_after_intrinsic_growth(unit, next_level):
    potion_first = deepcopy(unit)
    growth_first = deepcopy(unit)
    apply_all(potion_first)
    applied_a = apply_dynamic_stat_growth(potion_first, next_level)
    applied_b = apply_dynamic_stat_growth(growth_first, next_level)
    apply_all(growth_first)
    assert applied_a == applied_b
    assert {k: potion_first.get(k, 0) for k in STATS} == {
        k: growth_first.get(k, 0) for k in STATS
    }
    for potion in POTIONS:
        assert potion_first[potion["permanent_counter_key"]] == 2
