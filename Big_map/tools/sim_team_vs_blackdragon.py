"""Монте-Карло: отряд (Герцог 2 ур. + 2 Берсерка спереди, Колдун сзади) против Чёрного дракона.

Бой прогоняется через настоящий BattleEnv.step(): дракон (Mage) бьёт по площади всех,
инициатива/смерти/AoE считает движок. Герцог 2 ур. получается применением реальной
_apply_hero_levelup_bonuses к базовому Герцогу (1 ур.).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    BattleEnv, BATTLE_TEMPLATE_BY_NAME, _apply_hero_levelup_bonuses,
    DEFEND_ACTION_INDEX, WAIT_ACTION_INDEX, RED_POSITIONS, BLUE_POSITIONS,
)
from data_dicts_compact_lines import placeholder_unit


def fill_team(units: list, team: str, positions: list) -> list:
    """Дополняет команду placeholder-юнитами на все пустые слоты (нужно для env._obs)."""
    used = {u["position"] for u in units}
    full = list(units)
    for pos in positions:
        if pos not in used:
            full.append(placeholder_unit(team, pos))
    return full


def make_unit(name: str, team: str, pos: int, stand: str) -> dict:
    t = BATTLE_TEMPLATE_BY_NAME[name]
    u = {
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
    return u


def duke_level2(team: str, pos: int, stand: str) -> dict:
    u = make_unit("Герцог", team, pos, stand)  # базовый 1 ур.
    u["Level"] = 2
    _apply_hero_levelup_bonuses(u)  # 1->2 ур.: x1.1 урон/HP, +1 точн.
    return u


def build_blue() -> list:
    return [
        make_unit("Берсерк", "blue", 7, "ahead"),
        duke_level2("blue", 8, "ahead"),
        make_unit("Берсерк", "blue", 9, "ahead"),
        make_unit("Колдун", "blue", 11, "behind"),
    ]


def run_one(log: bool = False) -> tuple:
    env = BattleEnv(log_enabled=log)
    red = fill_team([make_unit("Чёрный дракон", "red", 2, "ahead")], "red", RED_POSITIONS)
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
    dragon = next(u for u in env.combined if u["team"] == "red" and u["name"] == "Чёрный дракон")
    dragon_hp_left = max(0, int(dragon.get("health", 0)))
    return env.winner, env.round_no, survivors, dragon_hp_left


def main():
    print("=== характеристики ===")
    d2 = duke_level2("blue", 8, "ahead")
    print(f"Герцог 2 ур.: HP={d2['max_health']} урон={d2['damage']} точн={d2['accuracy']}% "
          f"инит={d2['initiative_base']} {d2['unit_type']}/{d2['attack_type_primary']}")
    for n in ("Берсерк", "Колдун", "Чёрный дракон"):
        t = BATTLE_TEMPLATE_BY_NAME[n]
        print(f"{n}: HP={t['max_health']} урон={t['damage']} точн={t['accuracy']}% "
              f"инит={t['initiative_base']} {t['unit_type']}/{t['attack_type_primary']} "
              f"имм={t['immunity']}")

    N = 3000
    wins = {"blue": 0, "red": 0, "draw": 0}
    rounds_list, surv_counts, dragon_dmg = [], [], []
    surv_by_unit = {"Берсерк": 0, "Герцог": 0, "Колдун": 0}
    for _ in range(N):
        w, r, surv, dhp = run_one(log=False)
        wins[w if w in wins else "draw"] += 1
        rounds_list.append(r)
        surv_counts.append(len(surv))
        dragon_dmg.append(800 - dhp)
        for name in surv:
            key = "Герцог" if name == "Герцог" else name
            surv_by_unit[key] = surv_by_unit.get(key, 0) + 1

    print(f"\n=== {N} боёв ===")
    print(f"Победы отряда (blue): {wins['blue']} ({100*wins['blue']/N:.1f}%)")
    print(f"Победы дракона (red): {wins['red']} ({100*wins['red']/N:.1f}%)")
    print(f"Ничьи/лимит:          {wins['draw']}")
    print(f"Средняя длительность: {sum(rounds_list)/N:.2f} раундов")
    print(f"Средний урон по дракону: {sum(dragon_dmg)/N:.0f} из 800 HP "
          f"({100*sum(dragon_dmg)/N/800:.0f}%)")
    print(f"Лучший бой (макс. урон по дракону): {max(dragon_dmg)} из 800 HP")
    print(f"Среднее число выживших в отряде: {sum(surv_counts)/N:.2f} из 4")
    print("Доля побед, где юнит выжил (по всем боям, берсерков 2 шт):")
    for k, v in surv_by_unit.items():
        denom = 2 * N if k == "Берсерк" else N
        print(f"  {k}: {100*v/denom:.1f}%")


if __name__ == "__main__":
    main()
