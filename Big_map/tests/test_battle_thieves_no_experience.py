import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, THIEF_UNIT_NAMES
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


EXPECTED_THIEF_NAMES = frozenset(
    {
        "Вор империи",
        "Вор легионов",
        "Вор гномов",
        "Вор нежити",
        "Вор эльфов",
    }
)


def _unit(name: str, position: int = 7) -> dict:
    source = next(entry for entry in DATA if entry.get("кто") == name)
    return map_unit_to_battle(source, team="blue", position=position)


def _enemy(exp_kill: int) -> dict:
    return {
        "name": "Test enemy",
        "team": "red",
        "position": 1,
        "health": 0,
        "max_health": 1,
        "running_away": 0,
        "exp_kill": int(exp_kill),
    }


def test_all_faction_thieves_are_recognized_and_receive_no_battle_exp():
    assert THIEF_UNIT_NAMES == EXPECTED_THIEF_NAMES

    data_names = {str(entry.get("кто", "") or "") for entry in DATA}
    assert EXPECTED_THIEF_NAMES <= data_names

    for thief_name in sorted(EXPECTED_THIEF_NAMES):
        thief = _unit(thief_name)
        before = deepcopy(thief)
        env = BattleEnv(log_enabled=False)
        env.combined = [thief, _enemy(exp_kill=10_000)]

        env._apply_battle_exp("red")

        assert thief["exp_current"] == before["exp_current"]
        assert thief["exp_required"] == before["exp_required"]
        assert thief["Level"] == before["Level"]
        assert thief["health"] == before["health"]
        assert thief["max_health"] == before["max_health"]
        assert thief["damage"] == before["damage"]
        assert thief["accuracy"] == before["accuracy"]
        assert env.last_levelups == []


def test_thief_is_excluded_without_reducing_other_winners_exp_share():
    thief = _unit("Вор империи", position=7)
    ally = _unit("Защитник Веры", position=8)
    ally["exp_current"] = 0
    env = BattleEnv(log_enabled=False)
    env.combined = [thief, ally, _enemy(exp_kill=20)]

    env._apply_battle_exp("red")

    assert thief["exp_current"] == 0
    assert ally["exp_current"] == 20


def test_thief_non_progression_persists_back_to_campaign():
    campaign = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    campaign.reset(seed=123)

    thief = _unit("Вор империи")
    thief["exp_current"] = int(thief["exp_required"]) - 1
    before = deepcopy(thief)
    for index, unit in enumerate(campaign.blue_team_state):
        if int(unit.get("position", -1)) == 7:
            campaign.blue_team_state[index] = deepcopy(thief)
            break

    battle_blue = deepcopy(campaign.blue_team_state)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [*battle_blue, _enemy(exp_kill=10_000)]
    battle._apply_battle_exp("red")
    campaign.battle_env = battle

    campaign._save_blue_state()

    saved = next(
        unit for unit in campaign.blue_team_state if int(unit.get("position", -1)) == 7
    )
    assert saved["name"] == "Вор империи"
    assert saved["Level"] == before["Level"]
    assert saved["exp_current"] == before["exp_current"]
    assert saved["exp_required"] == before["exp_required"]
    assert saved["max_health"] == before["max_health"]
    assert saved["damage"] == before["damage"]
    assert saved["accuracy"] == before["accuracy"]
