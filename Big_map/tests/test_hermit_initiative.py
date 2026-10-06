import unittest
from copy import deepcopy

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


UNITS = {unit["кто"]: unit for unit in DATA}


class HermitInitiativeTests(unittest.TestCase):
    def setUp(self):
        self.battle = BattleEnv()
        self.hermit = map_unit_to_battle(UNITS["Отшельник"], "red", 6)

    def victim(self, initiative, team="blue"):
        unit = map_unit_to_battle(UNITS["Демон"], team, 8 if team == "blue" else 2)
        unit.update(initiative_base=initiative, initiative=initiative, resistance=[])
        self.battle.combined = [self.hermit, unit]
        return unit

    def test_natural_recovery_restores_exact_value_on_both_sides(self):
        for team in ("red", "blue"):
            for initiative in (0, 1, 35, 45, 55, 60, 65, 101):
                with self.subTest(team=team, initiative=initiative):
                    unit = self.victim(initiative, team)
                    self.assertTrue(self.battle._apply_hermit_initiative_slow(self.hermit, unit))
                    self.battle.rng.seed(1)  # First recovery roll succeeds.
                    self.battle._apply_start_of_turn_effects(unit)
                    self.assertEqual(unit["initiative_base"], initiative)
                    self.assertEqual(unit["hermited"], 0)
                    self.assertNotIn("hermit_original_initiative_base", unit)

    def test_cleansing_and_repeated_applications_do_not_drift(self):
        for initiative in (35, 65):
            unit = self.victim(initiative)
            for _ in range(5):
                self.assertTrue(self.battle._apply_hermit_initiative_slow(self.hermit, unit))
                self.assertFalse(self.battle._apply_hermit_initiative_slow(self.hermit, unit))
                self.assertEqual(unit["hermit_original_initiative_base"], initiative)
                self.assertTrue(self.battle._cleanse_negative_effects(unit))
                self.assertEqual(unit["initiative_base"], initiative)
                self.assertFalse(self.battle._restore_hermit_initiative(unit))
                self.assertNotIn("hermit_original_initiative_base", unit)

    def test_failed_recovery_preserves_original_value(self):
        unit = self.victim(35)
        self.battle._apply_hermit_initiative_slow(self.hermit, unit)
        self.battle.rng.seed(0)  # First recovery roll fails.
        self.battle._apply_start_of_turn_effects(unit)
        self.assertEqual(unit["initiative_base"], 18)
        self.assertEqual(unit["hermit_original_initiative_base"], 35)
        self.assertEqual(unit["hermited"], 1)

    def test_immunity_and_ward_do_not_create_saved_value(self):
        for field in ("immunity", "resistance"):
            with self.subTest(field=field):
                unit = self.victim(65)
                unit[field] = ["Water"]
                self.assertFalse(self.battle._apply_hermit_initiative_slow(self.hermit, unit))
                self.assertEqual(unit["initiative_base"], 65)
                self.assertNotIn("hermit_original_initiative_base", unit)

    def test_campaign_save_restores_active_slow_and_preserves_recovery(self):
        env = CampaignEnv(
            map_name="super_last_stand", use_boss_starting_roster=False,
            scripted_capital_bot_enabled=False, log_enabled=False
        )
        self.addCleanup(env.close)
        for initiative in (35, 65):
            for state in ("active", "recovered", "dead"):
                with self.subTest(initiative=initiative, state=state):
                    env.reset(seed=42)
                    team = env._create_starting_blue_team()
                    # Regular recruit for the persistence check; retain the map's hero.
                    recruit = self.victim(initiative)
                    team = [recruit if unit["position"] == 8 else unit for unit in team]
                    env.blue_team_state = deepcopy(team)
                    self.battle.combined = deepcopy(team)
                    unit = next(unit for unit in self.battle.combined if unit["position"] == 8)
                    self.battle._apply_hermit_initiative_slow(self.hermit, unit)
                    if state == "recovered":
                        self.battle._cleanse_negative_effects(unit)
                    elif state == "dead":
                        unit["health"] = 0
                    env.battle_env = self.battle
                    env._save_blue_state()
                    saved = next(unit for unit in env.blue_team_state if unit["position"] == 8)
                    self.assertEqual(saved["initiative_base"], initiative)
                    self.assertNotIn("hermit_original_initiative_base", saved)
                    self.assertNotIn("hermited", saved)
                    if state == "dead":
                        self.assertEqual(saved["health"], 0)


if __name__ == "__main__":
    unittest.main()
