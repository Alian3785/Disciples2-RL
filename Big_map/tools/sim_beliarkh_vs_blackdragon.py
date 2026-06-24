"""Монте-Карло: Белиарх (blue) против Чёрного дракона (red), 1 на 1.

Использует реальные примитивы движка BattleEnv (_roll_hit, _apply_damage_with_armor,
_is_immune_damage, _alive) и точные характеристики юнитов из BATTLE_TEMPLATE_BY_NAME.
Инициатива: Белиарх 60 > дракон 40 (джиттер +0..9 не пересекается) -> Белиарх всегда первый.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, BATTLE_TEMPLATE_BY_NAME


def make_unit(name: str, team: str, pos: int) -> dict:
    t = BATTLE_TEMPLATE_BY_NAME[name]
    return {
        "name": name, "team": team, "position": pos, "stand": t.get("stand", "ahead"),
        "unit_type": t["unit_type"], "health": t["max_health"], "max_health": t["max_health"],
        "damage": t["damage"], "damage_secondary": t["damage_secondary"],
        "initiative_base": t["initiative_base"], "accuracy": t["accuracy"],
        "accuracy_secondary": t["accuracy_secondary"],
        "attack_type_primary": t["attack_type_primary"],
        "attack_type_secondary": t["attack_type_secondary"],
        "immunity": list(t["immunity"]), "resistance": list(t["resistance"]),
        "armor": t["armor"], "big": t["big"],
    }


def attack(env, atk, vic):
    """Один удар по правилам движка. Возвращает нанесённый урон."""
    if not env._roll_hit(atk):
        return 0
    if env._is_immune_damage(atk, vic):
        return 0
    dmg = env._apply_damage_with_armor(atk, atk["damage"], vic)
    vic["health"] -= dmg
    return dmg


def main():
    for n in ("Белиарх", "Чёрный дракон"):
        t = BATTLE_TEMPLATE_BY_NAME[n]
        print(f"{n}: HP={t['max_health']} урон={t['damage']} точн={t['accuracy']}% "
              f"инит={t['initiative_base']} тип={t['unit_type']} "
              f"атака={t['attack_type_primary']} имм={t['immunity']} броня={t['armor']}")

    N = 3000
    wins = {"blue": 0, "red": 0, "draw": 0}
    rounds_list, dmg_to_dragon, dmg_by_beliarkh_hits = [], [], []
    for _ in range(N):
        env = BattleEnv(log_enabled=False)
        env._init_with_custom_teams([make_unit("Чёрный дракон", "red", 2)],
                                    [make_unit("Белиарх", "blue", 7)])
        b = next(u for u in env.combined if u["team"] == "blue")
        d = next(u for u in env.combined if u["team"] == "red")
        start_hp = d["health"]
        rnd = 0
        while env._alive(b) and env._alive(d) and rnd < 500:
            rnd += 1
            if env._alive(b):  # Белиарх первый (инициатива выше)
                attack(env, b, d)
            if env._alive(d):
                attack(env, d, b)
        if env._alive(b) and not env._alive(d):
            wins["blue"] += 1
        elif env._alive(d) and not env._alive(b):
            wins["red"] += 1
        else:
            wins["draw"] += 1
        rounds_list.append(rnd)
        dmg_to_dragon.append(start_hp - max(0, d["health"]))

    print(f"\nБоёв: {N}")
    print(f"Победы Белиарха (blue): {wins['blue']} ({100*wins['blue']/N:.2f}%)")
    print(f"Победы дракона (red):   {wins['red']} ({100*wins['red']/N:.2f}%)")
    print(f"Ничьи/лимит:            {wins['draw']}")
    print(f"Средняя длительность боя: {sum(rounds_list)/N:.2f} раундов")
    print(f"Средний урон Белиарха по дракону до гибели: {sum(dmg_to_dragon)/N:.1f} из 800 HP")
    print(f"Макс. урон по дракону за бой: {max(dmg_to_dragon)} из 800 HP")


if __name__ == "__main__":
    main()
