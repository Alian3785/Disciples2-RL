from campaign_env import CampaignEnv


def test_save_blue_state_restores_battle_start_armor():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env._init_battle(enemy_id=1)

    assert env.battle_env is not None

    unit = next(
        u
        for u in env.battle_env.combined
        if u.get("team") == "blue" and int(u.get("position", -1)) == 7
    )
    base_armor = int(unit.get("base_armor", unit.get("armor", 0)) or 0)

    # DEFEND is a damage modifier and must not alter persistent armor.
    unit["defense"] = 1

    env._save_blue_state()

    saved_unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 7)
    assert int(saved_unit.get("armor", -1)) == base_armor
    assert int(saved_unit.get("defense", -1)) == 0
