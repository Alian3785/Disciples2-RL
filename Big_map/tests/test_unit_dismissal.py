import unittest
from copy import deepcopy

from campaign_env import CampaignEnv
from maps import available_maps


class UnitDismissalTests(unittest.TestCase):
    def setUp(self):
        self.env = CampaignEnv(
            map_name="trade_train", scripted_capital_bot_enabled=False,
            use_boss_starting_roster=False, log_enabled=False,
            observation_version="baseline",
        )
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)

    def dismiss_action(self, position):
        return self.env.GRID_DISMISS_UNIT_ACTION_START + position - 7

    def recruit_cultists(self):
        index = next(i for i, option in enumerate(self.env.active_hire_options)
                     if option["name"] == "Сектант")
        action = self.env.GRID_HIRE_ACTION_START + index
        for _ in range(3):
            self.assertTrue(self.env.compute_action_mask()[action])
            self.assertTrue(self.env.step(action)[4]["hired"])
        return action

    def kill(self, position):
        unit = self.env._blue_unit_at_position(position)
        unit.update(health=0.0, hp=0.0)
        return unit

    def mercenary_action(self, name):
        index = next(i for i, option in enumerate(self.env.active_mercenary_hire_options)
                     if option["unit_name"] == name)
        return self.env.GRID_MERCENARY_HIRE_ACTION_START + index

    def set_party(self, companions, hero_position=8, hero_level=1):
        hero = deepcopy(self.env._resolve_travel_hero())
        hero.update(position=hero_position, Level=hero_level)
        units = [hero]
        for name, position in companions:
            data = self.env._find_unit_data_by_name(name)
            units.append(self.env._build_unit_from_data(data, "blue", position))
        self.env.blue_team_state = self.env._build_battle_team_with_placeholders("blue", units)
        self.env._mark_equipment_dirty()
        self.env._sync_hero_progression_flags()

    def test_death_reserves_leadership_and_blocks_both_kinds_of_hire(self):
        self.recruit_cultists()
        self.kill(10)
        composition = self.env._party_hire_composition()
        self.assertEqual(composition["occupied_capacity"], 3)
        self.assertEqual(composition["leadership_points"], 0)
        warrior = self.env.GRID_HIRE_ACTION_START
        self.assertFalse(self.env.compute_action_mask()[warrior])
        self.assertFalse(self.env.step(warrior)[4]["hired"])
        self.env.grid_env.agent_pos = self.env.mercenary_interaction_tiles[0]
        mercenary = self.mercenary_action("Защитник Веры")
        gold = self.env.gold
        self.assertFalse(self.env.compute_action_mask()[mercenary])
        self.assertFalse(self.env.step(mercenary)[4]["hired"])
        self.assertEqual(self.env.gold, gold)
        revive = self.env.GRID_REVIVE_ACTION_START + list(self.env.GRID_BOTTLE_POSITIONS).index(10)
        self.assertTrue(self.env.compute_action_mask()[revive])
        info = self.env.step(revive)[4]
        self.assertTrue(info["revived"])
        self.assertEqual(info["leadership_occupied"], info["leadership_capacity"])

    def test_dismiss_dead_unit_then_hire_mercenary_in_same_slot(self):
        self.recruit_cultists()
        self.kill(10)
        self.env.grid_env.agent_pos = self.env.mercenary_interaction_tiles[0]
        hire = self.mercenary_action("Белый маг")
        self.assertFalse(self.env.compute_action_mask()[hire])
        gold, moves, turns = self.env.gold, self.env.moves, self.env.turns
        info = self.env.step(self.dismiss_action(10))[4]
        self.assertTrue(info["dismissed"])
        self.assertFalse(info["dismissed_was_alive"])
        self.assertEqual(info["dismissed_leadership"], 1)
        self.assertEqual(info["leadership_points"], 1)
        self.assertEqual((self.env.gold, self.env.moves, self.env.turns), (gold, moves, turns))
        self.assertFalse(self.env.compute_action_mask()[self.dismiss_action(10)])
        self.assertFalse(self.env._can_revive_position(10))
        self.assertTrue(self.env.compute_action_mask()[hire])
        info = self.env.step(hire)[4]
        self.assertTrue(info["hired"])
        self.assertEqual(info["hired_unit_name"], "Белый маг")
        self.assertEqual(info["hired_position"], 10)
        self.assertEqual(info["leadership_occupied"], 3)

    def test_live_and_dead_companions_can_be_dismissed_in_each_of_six_positions(self):
        for position in range(7, 13):
            for alive in (True, False):
                with self.subTest(position=position, alive=alive):
                    self.env.reset(seed=42)
                    self.set_party([("Сектант", position)], hero_position=7 if position == 8 else 8)
                    if not alive:
                        self.kill(position)
                    action = self.dismiss_action(position)
                    self.assertTrue(self.env.compute_action_mask()[action])
                    obs, _, terminated, truncated, info = self.env.step(action)
                    self.assertTrue(info["dismissed"])
                    self.assertEqual(info["dismissed_was_alive"], alive)
                    self.assertFalse(terminated or truncated)
                    self.assertTrue(self.env.observation_space.contains(obs))
                    self.assertEqual(info["leadership_occupied"], 0)
                    self.assertTrue(self.env._is_blue_position_empty_for_hire(position))
                    self.assertFalse(self.env.compute_action_mask()[action])

    def test_empty_slots_and_hero_are_masked_and_direct_calls_do_nothing(self):
        for hero_position in range(7, 13):
            with self.subTest(hero_position=hero_position):
                self.env.reset(seed=42)
                self.set_party([], hero_position=hero_position)
                self.assertFalse(self.env.compute_action_mask()[-6:].any())
                before = deepcopy(self.env.blue_team_state)
                for position in range(7, 13):
                    info = self.env.step(self.dismiss_action(position))[4]
                    self.assertFalse(info["dismissed"])
                self.assertEqual(self.env.blue_team_state, before)

    def test_dead_leader_blocks_dismissal_of_living_and_dead_companions(self):
        self.recruit_cultists()
        self.kill(8)
        self.kill(10)
        self.assertFalse(self.env.compute_action_mask()[-6:].any())
        for position in (8, 10, 11):
            self.assertFalse(self.env.step(self.dismiss_action(position))[4]["dismissed"])

    def test_large_unit_reserves_two_points_even_when_dead_and_releases_both_cells(self):
        for alive in (True, False):
            for clicked_position in (7, 10):
                with self.subTest(alive=alive, clicked_position=clicked_position):
                    self.env.reset(seed=42)
                    self.set_party([("Демон", 7), ("Сектант", 12)])
                    self.assertTrue(self.env._is_big_blue_unit(self.env._blue_unit_at_position(7)))
                    if not alive:
                        self.kill(7)
                    self.assertEqual(self.env._available_leadership_points(), 0)
                    self.assertTrue(self.env.compute_action_mask()[self.dismiss_action(clicked_position)])
                    info = self.env.step(self.dismiss_action(clicked_position))[4]
                    self.assertTrue(info["dismissed"])
                    self.assertEqual(info["dismissed_leadership"], 2)
                    self.assertEqual(info["leadership_points"], 2)
                    for position in (7, 10):
                        self.assertTrue(self.env._is_blue_position_empty_for_hire(position))
                        self.assertFalse(self.env.compute_action_mask()[self.dismiss_action(position)])

    def test_dismissal_clears_potion_buffs_before_replacement(self):
        hire = self.recruit_cultists()
        self.env.active_haste_elixir_positions.update((10, 11))
        self.env.active_potion_effect_positions["test_effect"] = {10, 12}
        self.env.step(self.dismiss_action(10))
        self.assertEqual(self.env.active_haste_elixir_positions, {11})
        self.assertEqual(self.env.active_potion_effect_positions["test_effect"], {12})
        self.assertTrue(self.env.step(hire)[4]["hired"])
        self.assertNotIn(10, self.env.active_haste_elixir_positions)

    def test_repeated_dismiss_and_rehire_cannot_farm_hire_reward(self):
        hire = self.recruit_cultists()
        for _ in range(3):
            self.assertTrue(self.env.step(self.dismiss_action(10))[4]["dismissed"])
            info = self.env.step(hire)[4]
            self.assertTrue(info["hired"])
            self.assertEqual(info["hire_reward"], 0)
        self.env.reset(seed=42)
        self.assertGreater(self.env.step(hire)[4]["hire_reward"], 0)

    def test_new_actions_are_appended_and_reset_keeps_layout(self):
        self.assertEqual(self.env.GRID_DISMISS_UNIT_ACTION_START, 203)
        self.assertEqual(self.env.action_space.n, 209)
        self.assertEqual(self.env.observation_space.shape, (2743,))
        for frozen in (True, False):
            self.env.freeze_dynamic_action_layout = frozen
            self.env.reset(seed=42)
            self.assertEqual(self.env.action_space.n, 209)
            self.assertEqual(self.env.GRID_DISMISS_UNIT_ACTION_START,
                             self.env.GRID_SWAP_UNIT_ACTION_START + self.env.GRID_SWAP_UNIT_ACTION_COUNT)
            self.assertEqual(len(self.env.compute_action_mask()), self.env.action_space.n)

    def test_dismissal_is_masked_in_battle(self):
        self.recruit_cultists()
        self.env._init_battle(6)
        self.env.mode = self.env.MODE_BATTLE
        self.assertFalse(self.env.compute_action_mask()[-6:].any())
        for position in (10, 11, 12):
            self.assertIsNone(self.env._dismiss_blue_unit_target(position))

    def test_full_party_objective_still_requires_living_companions(self):
        self.set_party([("Сектант", p) for p in (7, 9, 10, 11, 12)], hero_level=6)
        self.env.campaign_objective = "full_party"
        self.kill(10)
        self.env._reset_full_party_objective_tracking()
        self.assertEqual(self.env._party_hire_composition()["occupied_capacity"], 5)
        info = {}
        _, terminated, _ = self.env._apply_full_party_objective_progress(
            reward=0, terminated=False, truncated=False, info=info)
        self.assertFalse(terminated)
        self.assertFalse(info["full_party_complete"])
        self.assertEqual(info["full_party_companion_capacity"], 4)
        self.assertTrue(self.env._revive_unit_at_position(10)[0])
        info = {}
        _, terminated, _ = self.env._apply_full_party_objective_progress(
            reward=0, terminated=False, truncated=False, info=info)
        self.assertTrue(terminated)
        self.assertTrue(info["full_party_complete"])


class AllMapDismissalTests(unittest.TestCase):
    def check_dismissal_on_map(self, map_name):
        for frozen in (True, False):
            env = CampaignEnv(
                map_name=map_name, scripted_capital_bot_enabled=False,
                use_boss_starting_roster=False, freeze_dynamic_action_layout=frozen,
                log_enabled=False,
            )
            try:
                for alive in (True, False):
                    with self.subTest(map=map_name, frozen=frozen, alive=alive):
                        env.reset(seed=42)
                        starting_roster = deepcopy(env.blue_team_state)
                        space_size = env.action_space.n
                        self.assertEqual(space_size, env.GRID_DISMISS_UNIT_ACTION_START + 6)
                        self.assertEqual(env.GRID_DISMISS_UNIT_ACTION_START,
                                         env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_ACTION_COUNT)
                        companion = next((u for u in env.blue_team_state
                                          if not env._is_empty_blue_unit(u) and not env._is_hero_unit(u)), None)
                        if companion is None:
                            # A controlled post-hire state for maps starting with only a hero.
                            position = env._find_first_empty_blue_position(tuple(range(7, 13)))
                            self.assertIsNotNone(position)
                            companion = env._build_unit_from_data(
                                env._find_unit_data_by_name("Сектант"), "blue", position)
                            env._replace_blue_unit(companion)
                            companion = env._blue_unit_at_position(position)
                        position = int(companion["position"])
                        if not alive:
                            companion.update(health=0.0, hp=0.0)
                        env._mark_equipment_dirty()
                        env._sync_hero_progression_flags()
                        occupied_before = env._party_hire_composition()["occupied_capacity"]
                        released = 2 if env._is_big_blue_unit(companion) else 1
                        mask = env.compute_action_mask()
                        self.assertEqual(len(mask), space_size)
                        for slot in env.GRID_DISMISS_UNIT_POSITIONS:
                            if env._is_blue_position_empty_for_hire(slot):
                                self.assertFalse(mask[env.GRID_DISMISS_UNIT_ACTION_START + slot - 7])
                        hero = env._resolve_travel_hero()
                        self.assertIsNotNone(hero)
                        self.assertFalse(mask[env.GRID_DISMISS_UNIT_ACTION_START + hero["position"] - 7])
                        action = env.GRID_DISMISS_UNIT_ACTION_START + position - 7
                        self.assertTrue(mask[action])
                        obs, _, _, _, info = env.step(action)
                        self.assertTrue(info["dismissed"])
                        self.assertEqual(info["dismissed_was_alive"], alive)
                        self.assertEqual(info["leadership_occupied"], occupied_before - released)
                        self.assertTrue(env._is_blue_position_empty_for_hire(position))
                        self.assertFalse(env.compute_action_mask()[action])
                        self.assertFalse(env._can_revive_position(position))
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertEqual(env.action_space.n, space_size)
                        env.reset(seed=42)
                        self.assertEqual(env.blue_team_state, starting_roster)
                        self.assertEqual(env.action_space.n, space_size)
            finally:
                env.close()


def _map_dismissal_test(map_name):
    def test(self):
        self.check_dismissal_on_map(map_name)
    return test


for _map_name in available_maps():
    setattr(AllMapDismissalTests, f"test_dismissal_{_map_name}", _map_dismissal_test(_map_name))


if __name__ == "__main__":
    unittest.main()
