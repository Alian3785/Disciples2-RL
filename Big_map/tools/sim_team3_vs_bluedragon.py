"""Монте-Карло: Герцог 3 ур. + 2 Тёмных паладина спереди, Демонолог сзади — против Синего дракона.

Бой прогоняется через настоящий BattleEnv.step() (дракон-Mage бьёт по площади всех).
Герцог 3 ур. = двойная прокачка базового Герцога (1->2->3): ×1.1 урон/HP за уровень, +1 точн.

Синий дракон: полное HP=700. По умолчанию стартует недобитым — на 140 меньше = 560 HP.
HP можно переопределить аргументом: python sim_team3_vs_bluedragon.py 560
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    BattleEnv, BATTLE_TEMPLATE_BY_NAME, _apply_hero_levelup_bonuses,
    DEFEND_ACTION_INDEX, WAIT_ACTION_INDEX, RED_POSITIONS, BLUE_POSITIONS,
)
from data_dicts_compact_lines import placeholder_unit

DRAGON_NAME = "Синий дракон"
DRAGON_FULL_HP = BATTLE_TEMPLATE_BY_NAME[DRAGON_NAME]["max_health"]  # 700
DRAGON_HP = int(sys.argv[1]) if len(sys.argv) > 1 else DRAGON_FULL_HP - 140  # 560


def make_unit(name: str, team: str, pos: int, stand: str) -> dict:
    t = BATTLE_TEMPLATE_BY_NAME[name]
    return {
        "name": name, "team": team, "position": pos, "stand": stand,
        "unit_type": t["unit_type"], "health": t["max_health"], "max_health": t["max_health"],
        "damage": t["damage"], "damage_secondary": t["damage_secondary"],
        "initiative_base": t["initiative_base"], "accuracy": t["accuracy"],
        "accuracy_secondary": t["accuracy_secondary"],
        "attack_type_primary": t["attack_type_primary"],
        "attack_type_secondary": t["attack_type_secondary"],
        "immunity": list(t["immunity"]), "resistance": list(t["resistance"]),
        "armor": t["armor"], "big": t["big"], "Level": t["Level"],
        "original_damage": t["damage"],
    }


def duke_level3(team: str, pos: int, stand: str) -> dict:
    u = make_unit("Герцог", team, pos, stand)  # базовый 1 ур.
    u["Level"] = 2
    _apply_hero_levelup_bonuses(u)  # 1 -> 2
    u["Level"] = 3
    _apply_hero_levelup_bonuses(u)  # 2 -> 3 (веха лидерства: +слот отряда, боевые статы не меняет)
    return u


def fill_team(units: list, team: str, positions: list) -> list:
    used = {u["position"] for u in units}
    return units + [placeholder_unit(team, p) for p in positions if p not in used]


def build_blue() -> list:
    return [
        make_unit("Темный паладин", "blue", 7, "ahead"),
        duke_level3("blue", 8, "ahead"),
        make_unit("Темный паладин", "blue", 9, "ahead"),
        make_unit("Демонолог", "blue", 11, "behind"),
    ]


def run_one(log: bool = False) -> tuple:
    env = BattleEnv(log_enabled=log)
    dragon0 = make_unit(DRAGON_NAME, "red", 2, "ahead")
    dragon0["health"] = int(DRAGON_HP)  # стартовое (недобитое) HP дракона
    red = fill_team([dragon0], "red", RED_POSITIONS)
    blue = fill_team(build_blue(), "blue", BLUE_POSITIONS)
    env._init_with_custom_teams(red, blue)
    done, steps = False, 0
    while not done and steps < 4000:
        steps += 1
        mask = env.compute_action_mask()
        dpos = next((u["position"] for u in env.combined
                     if u["team"] == "red" and env._alive(u)), None)
        if dpos is not None and mask[dpos - 1]:
            action = dpos - 1
        else:
            tgt = [i for i in range(12) if mask[i]]
            action = tgt[0] if tgt else (DEFEND_ACTION_INDEX if mask[DEFEND_ACTION_INDEX]
                                         else WAIT_ACTION_INDEX)
        _, _, done, _, _ = env.step(action)
    survivors = [u["name"] for u in env.combined if u["team"] == "blue" and env._alive(u)]
    dragon = next(u for u in env.combined if u["team"] == "red" and u["name"] == DRAGON_NAME)
    return env.winner, env.round_no, survivors, max(0, int(dragon.get("health", 0)))


def main():
    print("=== характеристики ===")
    d3 = duke_level3("blue", 8, "ahead")
    print(f"Герцог 3 ур.: HP={d3['max_health']} урон={d3['damage']} точн={d3['accuracy']}% "
          f"инит={d3['initiative_base']} {d3['unit_type']}/{d3['attack_type_primary']}")
    for n in ("Темный паладин", "Демонолог", DRAGON_NAME):
        t = BATTLE_TEMPLATE_BY_NAME[n]
        print(f"{n}: HP={t['max_health']} урон={t['damage']} точн={t['accuracy']}% "
              f"инит={t['initiative_base']} {t['unit_type']}/{t['attack_type_primary']} имм={t['immunity']}")
    print(f"\nСтартовое HP дракона в бою: {DRAGON_HP} из {DRAGON_FULL_HP} "
          f"(недобит на {DRAGON_FULL_HP - DRAGON_HP})")

    N = 3000
    wins = {"blue": 0, "red": 0, "draw": 0}
    rounds_list, surv_counts, dragon_dmg = [], [], []
    surv_by_unit = {"Темный паладин": 0, "Герцог": 0, "Демонолог": 0}
    for _ in range(N):
        w, r, surv, dhp = run_one(log=False)
        wins[w if w in wins else "draw"] += 1
        rounds_list.append(r)
        surv_counts.append(len(surv))
        dragon_dmg.append(DRAGON_HP - dhp)
        for name in surv:
            surv_by_unit[name] = surv_by_unit.get(name, 0) + 1

    print(f"\n=== {N} боёв ===")
    print(f"Победы отряда (blue): {wins['blue']} ({100*wins['blue']/N:.1f}%)")
    print(f"Победы дракона (red): {wins['red']} ({100*wins['red']/N:.1f}%)")
    print(f"Ничьи/лимит:          {wins['draw']}")
    print(f"Средняя длительность: {sum(rounds_list)/N:.2f} раундов")
    print(f"Средний урон по дракону: {sum(dragon_dmg)/N:.0f} из {DRAGON_HP} стартовых "
          f"({100*sum(dragon_dmg)/N/DRAGON_HP:.0f}%)")
    print(f"Лучший бой (макс. урон по дракону): {max(dragon_dmg)} из {DRAGON_HP}")
    print(f"Среднее число выживших: {sum(surv_counts)/N:.2f} из 4")
    print("Доля боёв, где юнит выжил (паладинов 2 шт):")
    for k, v in surv_by_unit.items():
        denom = 2 * N if k == "Темный паладин" else N
        print(f"  {k}: {100*v/denom:.1f}%")


if __name__ == "__main__":
    main()
