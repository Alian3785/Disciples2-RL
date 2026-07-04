"""Регрессионные тесты бага №1: призыв юнитов типа Summoner (регистр unit_type).

Также покрывают связанные решения:
- призванные юниты не дают опыта за убийство (анти-фарм);
- опыт трупа, затёртого призывом, не теряется победителем.
"""

import random

import pytest

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from summon_actions import handle_occultmaster_action


def _entry(name: str) -> dict:
    for e in DATA:
        if isinstance(e, dict) and str(e.get("кто", "")).strip() == name:
            return e
    raise KeyError(name)


def _make_env(red_placement: dict, blue_placement: dict, seed: int = 0) -> BattleEnv:
    env = BattleEnv(log_enabled=True)
    env.rng = random.Random(seed)
    red = [
        map_unit_to_battle(_entry(red_placement[pos]), "red", pos)
        if pos in red_placement
        else placeholder_unit("red", pos)
        for pos in range(1, 7)
    ]
    blue = [
        map_unit_to_battle(_entry(blue_placement[pos]), "blue", pos)
        if pos in blue_placement
        else placeholder_unit("blue", pos)
        for pos in range(7, 13)
    ]
    env._init_with_custom_teams(red, blue)
    return env


def test_is_summoner_type_case_insensitive():
    assert BattleEnv._is_summoner_type("Summoner")
    assert BattleEnv._is_summoner_type("summoner")
    assert BattleEnv._is_summoner_type(" SUMMONER ")
    assert not BattleEnv._is_summoner_type("Occultmaster")
    assert not BattleEnv._is_summoner_type("")
    assert not BattleEnv._is_summoner_type(None)


def test_blue_summoner_mask_targets_own_empty_cells():
    env = _make_env({3: "Грифон"}, {8: "Элементалист"})
    mask = env.compute_action_mask()
    # Конвенция маски призывателя зеркальная (как лечение Патриарха):
    # действие на красную клетку означает призыв в свою клетку напротив.
    assert bool(mask[2]), "red pos3 ↔ своя пустая pos9 — призыв должен быть доступен"
    assert not bool(mask[1]), "red pos2 ↔ своя pos8 занята самим Элементалистом"
    assert not any(bool(mask[i]) for i in range(6, 12)), (
        "прямые действия на синие клетки призывателю недоступны"
    )


def test_blue_elementalist_summons_air_elemental():
    env = _make_env({3: "Грифон"}, {8: "Элементалист"})
    env.step(2)  # зеркало red pos3 -> призыв на свою pos9
    summoned = [
        u
        for u in env.combined
        if u.get("name") == "Элементаль Воздуха" and u.get("team") == "blue"
    ]
    assert len(summoned) == 1
    assert int(summoned[0].get("position")) == 9
    assert summoned[0].get("Summoned") == 8
    assert float(summoned[0].get("exp_kill", -1)) == 0


def test_red_sage_summons_ent_via_scripted_ai():
    # Мудрец (иниц 40) ходит раньше Служки (иниц 10): к первому ходу BLUE
    # скриптованный ИИ красных уже должен был призвать энта.
    env = _make_env({5: "Мудрец"}, {12: "Служка"})
    ents = [
        u
        for u in env.combined
        if u.get("name") == "Малый энт" and u.get("team") == "red"
    ]
    assert ents, "красный Мудрец должен призвать Малого энта"
    assert all(float(u.get("exp_kill", -1)) == 0 for u in ents)


def test_overwritten_corpse_exp_counted(monkeypatch):
    env = _make_env({5: "Мастер оккультист"}, {8: "Титан"})
    corpse = env._unit_by_position(2)
    corpse.clear()
    corpse.update(
        {
            "name": "Ониксовая гаргулья",
            "team": "red",
            "position": 2,
            "stand": "ahead",
            "health": 0,
            "max_health": 170,
            "big": False,
            "exp_kill": 810,
        }
    )
    monkeypatch.setattr("summon_actions.random.choice", lambda variants: variants[0])

    occ = env._unit_by_position(5)
    spawned, _reason = handle_occultmaster_action(
        env, occ, [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
    )

    assert spawned is True
    assert env.overwritten_exp_kill["red"] == pytest.approx(810.0)

    # Труп затёрт призывом, но при подсчёте опыта за бой он учитывается:
    # Мастер оккультист (1085) + затёртая гаргулья (810); призванные дают 0.
    env._apply_battle_exp("red")
    assert env.last_battle_exp == pytest.approx(1085.0 + 810.0)


def test_hero_item_summon_overwrite_preserves_corpse_exp():
    env = _make_env({3: "Грифон"}, {8: "Мастер клинка"})
    corpse = env._unit_by_position(9)
    corpse.clear()
    corpse.update(
        {
            "name": "Ониксовая гаргулья",
            "team": "blue",
            "position": 9,
            "stand": "ahead",
            "health": 0,
            "max_health": 170,
            "big": False,
            "exp_kill": 810,
        }
    )
    applied = env._apply_hero_item_summon(
        source_unit=env._unit_by_position(8),
        target_team="blue",
        target_pos=9,
        target_unit=corpse,
        effect={"summon_unit_name": "Малый энт"},
    )
    assert applied == 1.0
    summoned = env._unit_by_position(9)
    assert summoned.get("name") == "Малый энт"
    assert float(summoned.get("exp_kill", -1)) == 0
    assert env.overwritten_exp_kill["blue"] == pytest.approx(810.0)
