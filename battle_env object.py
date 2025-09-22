# ============================================================
# Gymnasium Environment: "RED vs BLUE" battle
# - Action space = 6 (targets pos1..pos6)
# - Includes big-units, poison/burn effects, accuracy2
# This module only defines data and the env. No training or viz here.
# ============================================================

from __future__ import annotations

import random
from operator import itemgetter
from typing import List, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
#5555555
#4y3443hjerjdfjdsfjsdfj
# --- encoding vocabularies ---
TYPE_LIST = [
    "Archer", "gargoil", "Mage", "Воин", "Demon", "Death", "lord", "Dead dragon"
]
ATTACK_TYPES = ["Weapon", "earth", "Fire", "poison", "death", "Mind"]


def _one_hot(value, vocab):
    return [1.0 if value == v else 0.0 for v in vocab]


def _multi_hot(values, vocab):
    values_set = set(values or [])
    return [1.0 if v in values_set else 0.0 for v in vocab]


class BaseUnit:
    def __init__(
        self,
        name,
        initiative,
        base_initiative,
        team,
        position,
        stand,
        unit_type,
        damage,
        damage2,
        health,
        maxhealth,
        armor,
        accuracy,
        accuracy2,
        immunity,
        resilience,
        attack_type1,
        attack_type2,
        big,
    ):
        self.name = name
        self.initiative = initiative
        self.base_initiative = base_initiative
        self.team = team
        self.position = position
        self.stand = stand
        self.unit_type = unit_type
        self.damage = damage
        self.damage2 = damage2
        self.health = health
        self.maxhealth = maxhealth
        self.armor = armor
        self.accuracy = accuracy
        self.accuracy2 = accuracy2
        self.immunity = list(immunity or [])
        self.resilience = list(resilience or [])
        self.attack_type1 = attack_type1 or "Weapon"
        self.attack_type2 = attack_type2 or ""
        self.big = bool(big)

        # runtime battle fields
        # Twohits: 1 for Demon (has a second strike available this round), else 0
        self.Twohits = 1 if unit_type == "Demon" else 0
        self.Runningaway = 0
        self.poison_turns_left = 0
        self.poison_damage_per_tick = 0
        self.burn_turns_left = 0
        self.burn_damage_per_tick = 0
        self.resilience_used_types = []

    def is_alive(self):
        return self.health > 0


# ------------------------ Units (prototypes) ------------------------
UNITS_RED = [
    BaseUnit(name="скелет-рыцарь", initiative=50, base_initiative=50, team="red", position=1, stand="ahead",
             unit_type="Воин", damage=100, damage2=0, health=270, maxhealth=270, armor=0, accuracy=80, accuracy2=0,
             immunity=["death", "Poison"], resilience=[], attack_type1="Weapon", attack_type2="", big=False),

    BaseUnit(name="рыцарь смерти", initiative=50, base_initiative=50, team="red", position=2, stand="ahead",
             unit_type="Воин", damage=120, damage2=0, health=255, maxhealth=255, armor=0, accuracy=87, accuracy2=0,
             immunity=["death"], resilience=[], attack_type1="Weapon", attack_type2="", big=False),

    BaseUnit(name="Мертвый дракон", initiative=35, base_initiative=35, team="red", position=3, stand="ahead",
             unit_type="Dead dragon", damage=65, damage2=20, health=450, maxhealth=450, armor=0, accuracy=80, accuracy2=40,
             immunity=["death"], resilience=[], attack_type1="death", attack_type2="poison", big=True),

    BaseUnit(name="Архилич", initiative=40, base_initiative=40, team="red", position=4, stand="behind",
             unit_type="Mage", damage=90, damage2=0, health=170, maxhealth=170, armor=0, accuracy=80, accuracy2=0,
             immunity=["death"], resilience=[], attack_type1="earth", attack_type2="", big=False),

    BaseUnit(name="Смерть1", initiative=40, base_initiative=40, team="red", position=5, stand="behind",
             unit_type="Death", damage=100, damage2=20, health=125, maxhealth=125, armor=0, accuracy=80, accuracy2=50,
             immunity=["Weapon", "death"], resilience=[], attack_type1="Weapon", attack_type2="poison", big=False),

    BaseUnit(name="пусто", initiative=0, base_initiative=0, team="red", position=6, stand="behind",
             unit_type="Archer", damage=0, damage2=0, health=0, maxhealth=0, armor=0, accuracy=0, accuracy2=0,
             immunity=["death"], resilience=[], attack_type1="Weapon", attack_type2="", big=False),
]

UNITS_BLUE = [
    BaseUnit(name="Астерот", initiative=50, base_initiative=50, team="blue", position=7, stand="ahead",
             unit_type="Demon", damage=150, damage2=0, health=1020, maxhealth=1020, armor=0, accuracy=80, accuracy2=0,
             immunity=[], resilience=["Mind", "Fire"], attack_type1="Weapon", attack_type2="", big=True),

    BaseUnit(name="Герцог", initiative=50, base_initiative=50, team="blue", position=8, stand="ahead",
             unit_type="Воин", damage=120, damage2=0, health=306, maxhealth=306, armor=0, accuracy=100, accuracy2=0,
             immunity=[], resilience=[], attack_type1="Weapon", attack_type2="", big=False),

    BaseUnit(name="Гаргулья", initiative=60, base_initiative=60, team="blue", position=9, stand="ahead",
             unit_type="gargoil", damage=85, damage2=0, health=170, maxhealth=170, armor=65, accuracy=80, accuracy2=0,
             immunity=["poison"], resilience=["Mind"], attack_type1="earth", attack_type2="", big=True),

    BaseUnit(name="пусто", initiative=0, base_initiative=0, team="blue", position=10, stand="behind",
             unit_type="Archer", damage=0, damage2=0, health=0, maxhealth=0, armor=0, accuracy=0, accuracy2=0,
             immunity=[], resilience=[], attack_type1="Weapon", attack_type2="", big=False),

    BaseUnit(name="пусто", initiative=0, base_initiative=0, team="blue", position=11, stand="behind",
             unit_type="Archer", damage=0, damage2=0, health=0, maxhealth=0, armor=0, accuracy=0, accuracy2=0,
             immunity=[], resilience=[], attack_type1="Weapon", attack_type2="", big=False),

    BaseUnit(name="пусто", initiative=0, base_initiative=0, team="blue", position=12, stand="behind",
             unit_type="Archer", damage=0, damage2=0, health=0, maxhealth=0, armor=0, accuracy=0, accuracy2=0,
             immunity=[], resilience=[], attack_type1="Weapon", attack_type2="", big=False),
]

# Position ranges
RED_POSITIONS:  List[int] = list(range(1, 7))
BLUE_POSITIONS: List[int] = list(range(7, 13))

# Available targets for the agent — enemies only (pos1..pos6)
TARGET_POSITIONS: List[int] = RED_POSITIONS

# Constants for observation normalization
MAX_HP   = max([u.health for u in (UNITS_RED + UNITS_BLUE)])
MAX_INIT = max([u.base_initiative for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG  = max([u.damage for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG2 = max([u.damage2 for u in (UNITS_RED + UNITS_BLUE)])

POISON_TURNS  = 3
BURN_DAMAGE   = 10
BURN_TURNS    = 3


class BattleEnv(gym.Env):
    """
    Observation (540): 12 slots × 45 features (includes Accuracy2).
    Action space: Discrete(6) — choose a target among enemies (pos1..pos6).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        penalty_invalid_target: float = -0.10,
        penalty_unreachable_warrior: float = -0.20,
        log_enabled: bool = False,
    ):
        super().__init__()
        self.reward_win  = float(reward_win)
        self.reward_loss = float(reward_loss)
        self.reward_step = float(reward_step)

        self.penalty_invalid_target = float(penalty_invalid_target)
        self.penalty_unreachable_warrior = float(penalty_unreachable_warrior)

        self.log_enabled = bool(log_enabled)
        self._pretty_events: List[str] = []

        # Action space = 6
        self.action_space = spaces.Discrete(len(TARGET_POSITIONS))

        # Observation space
        low_unit = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0]  # HP, INI, INI_base, DMG, DMG2, team, pos, stand
            + [0]*len(TYPE_LIST)       # type one-hot
            + [0]*len(ATTACK_TYPES)    # immunity
            + [0]*len(ATTACK_TYPES)    # atk1
            + [0]*len(ATTACK_TYPES)    # atk2
            + [0]*len(ATTACK_TYPES)    # resilience
            + [0]                      # armor
            + [0]                      # accuracy
            + [0]                      # accuracy2
            + [0, 0],                  # poison_left, burn_left
            dtype=np.float32,
        )
        high_unit = np.array(
            [MAX_HP, MAX_INIT, MAX_INIT, MAX_DMG, MAX_DMG2, 1, 12, 1]
            + [1]*len(TYPE_LIST)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [100]         # armor
            + [100]         # accuracy
            + [100]         # accuracy2
            + [POISON_TURNS, BURN_TURNS],
            dtype=np.float32,
        )
        low  = np.tile(low_unit, 12)
        high = np.tile(high_unit, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Battle state
        self.rng = random.Random()
        self.combined = []
        self.round_no: int = 1
        self.winner: Optional[str] = None
        self.current_blue_attacker_pos: Optional[int] = None
        self._lord_applied_burn: Dict[int, bool] = {}

    # ------------------ Helpers ------------------
    def seed(self, seed=None):
        self.rng = random.Random(seed)

    def _alive(self, u):
        return u.is_alive()

    def _team_alive(self, team):
        return any(self._alive(u) and u.team == team for u in self.combined)

    def _unit_by_position(self, pos: int):
        return next((u for u in self.combined if u.position == pos), None)

    def _live_positions_of(self, team: str):
        return [u.position for u in self.combined if u.team == team and self._alive(u)]

    def _log(self, s: str):
        if self.log_enabled:
            self._pretty_events.append(s)

    def pop_pretty_events(self):
        out = self._pretty_events[:]
        self._pretty_events.clear()
        return out

    def _col_of(self, pos):
        return (pos - 1) % 3

    def _enemy_rows(self, team: str):
        if team == "red":
            return [7, 8, 9], [10, 11, 12]
        else:
            return [1, 2, 3], [4, 5, 6]

    def _warrior_near_far_cols(self, col: int):
        if col == 0:
            return [0, 1], [2]
        if col == 1:
            return [0, 1, 2], []
        return [1, 2], [0]

    def _warrior_allowed_targets(self, attacker):
        assert attacker.unit_type in ("Воин", "Demon", "lord")
        if attacker.stand == "behind":
            if any(
                self._alive(u)
                and u.team == attacker.team
                and u.unit_type in ("Воин", "Demon", "lord")
                and u.stand == "ahead"
                for u in self.combined
            ):
                return []
        ahead, behind = self._enemy_rows(attacker.team)
        col = self._col_of(attacker.position)
        near_cols, far_cols = self._warrior_near_far_cols(col)

        def alive(pos):
            uu = self._unit_by_position(pos)
            return uu is not None and self._alive(uu)

        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near:
            return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far:
            return far

        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2:
            return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

    def _pick_lowest_hp(self, positions):
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u.health, p))
        if not candidates:
            return None
        min_hp = min(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == min_hp]
        return self.rng.choice(best)

    # ----------- start-of-turn effects (poison/burn) -----------
    def _apply_start_of_turn_effects(self, unit):
        # Poison
        if unit.poison_turns_left > 0 and self._alive(unit):
            if "poison" in (unit.immunity or []):
                unit.poison_turns_left = 0
                unit.poison_damage_per_tick = 0
                self._log(
                    f"🛡 Иммунитет к эффекту 'poison' — яд не действует на {unit.team.upper()} {unit.name}#{unit.position}."
                )
            else:
                tick = int(unit.poison_damage_per_tick or 0)
                if tick > 0:
                    before = unit.health
                    unit.health -= tick
                    unit.poison_turns_left -= 1
                    after = unit.health
                    self._log(
                        f"☠ Яд поражает {unit.team.upper()} {unit.name}#{unit.position}: "
                        f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit.poison_turns_left}"
                    )
                    if unit.health <= 0:
                        unit.initiative = 0
                        self._log(f"✖ {unit.team.upper()} {unit.name}#{unit.position} погибает от яда.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit.poison_turns_left -= 1
                if unit.poison_turns_left <= 0:
                    unit.poison_damage_per_tick = 0

        # Burn — tick damage from burn_damage_per_tick (for lord = его урон2)
        if unit.burn_turns_left > 0 and self._alive(unit):
            if "Fire" in (unit.immunity or []):
                unit.burn_turns_left = 0
                unit.burn_damage_per_tick = 0
                self._log(
                    f"🛡 Иммунитет к эффекту 'Fire' — поджог не действует на {unit.team.upper()} {unit.name}#{unit.position}."
                )
            else:
                tick = int(unit.burn_damage_per_tick or BURN_DAMAGE)
                if tick > 0:
                    before = unit.health
                    unit.health -= tick
                    unit.burn_turns_left -= 1
                    after = unit.health
                    self._log(
                        f"🔥 Поджог поражает {unit.team.upper()} {unit.name}#{unit.position}: "
                        f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit.burn_turns_left}"
                    )
                    if unit.health <= 0:
                        unit.initiative = 0
                        self._log(f"✖ {unit.team.upper()} {unit.name}#{unit.position} погибает от поджога.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit.burn_turns_left -= 1
                if unit.burn_turns_left <= 0:
                    unit.burn_damage_per_tick = 0

        return True

    # ------------------ Core battle logic ------------------
    def _reset_state(self):
        # Строим новые экземпляры каждый раз, чтобы не переносить состояние между эпизодами
        self.combined = []
        for proto in (UNITS_RED + UNITS_BLUE):
            self.combined.append(
                BaseUnit(
                    name=proto.name,
                    initiative=proto.base_initiative,
                    base_initiative=proto.base_initiative,
                    team=proto.team,
                    position=proto.position,
                    stand=proto.stand,
                    unit_type=proto.unit_type,
                    damage=proto.damage,
                    damage2=proto.damage2,
                    health=proto.health,
                    maxhealth=proto.maxhealth,
                    armor=proto.armor,
                    accuracy=proto.accuracy,
                    accuracy2=proto.accuracy2,
                    immunity=list(proto.immunity),
                    resilience=list(proto.resilience),
                    attack_type1=proto.attack_type1,
                    attack_type2=proto.attack_type2,
                    big=proto.big,
                )
            )
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self._lord_applied_burn = {}
        self._log(f"Эпизод начат. Раунд {self.round_no}.")

    def _candidates(self):
        return [u for u in self.combined if self._alive(u) and u.initiative > 0]

    def _pop_next(self):
        cand = self._candidates()
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=lambda x: x.initiative, reverse=True)
        return cand[0]

    def _end_round_restore(self):
        for u in self.combined:
            u.initiative = u.base_initiative if self._alive(u) else 0
            # В конце раунда демоны восстанавливают возможность второго удара
            if u.unit_type == "Demon":
                u.Twohits = 1
        self._log("Восстановление инициативы. Новый раунд.")

    def _is_immune_damage(self, attacker, victim):
        return attacker.attack_type1 in (victim.immunity or [])

    def _is_immune_status(self, attacker, victim):
        atk2 = attacker.attack_type2
        return bool(atk2) and (atk2 in (victim.immunity or []))

    def _resilience_blocks(self, attacker, victim):
        res_list = set(victim.resilience or [])
        if not res_list:
            return False
        used = set(victim.resilience_used_types or [])
        available = res_list - used
        if not available:
            return False
        atk1 = attacker.attack_type1
        atk2 = attacker.attack_type2
        for tag in (atk1, atk2):
            if tag and tag in available:
                victim.resilience_used_types.append(tag)
                self._log(
                    f"🧿 Стойкость — первый удар типа '{tag}' по "
                    f"{victim.team.upper()} {victim.name}#{victim.position} поглощён."
                )
                return True
        return False

    def _apply_damage_with_armor(self, base_dmg, victim):
        armor = float(victim.armor or 0)
        factor = (1.0 - armor / 100.0) if armor > 0 else 1.0
        eff = base_dmg * max(0.0, factor)
        return int(round(eff))

    def _roll_hit(self, attacker):
        acc = float(attacker.accuracy or 100)
        return self.rng.random() < (acc / 100.0)

    def _roll_status(self, chance_percent):
        cp = max(0.0, min(100.0, float(chance_percent or 0.0)))
        return self.rng.random() < (cp / 100.0)

    def _attack(self, attacker, target_pos):
        # AoE for Mage and Dead dragon
        if attacker is not None and attacker.unit_type in ("Mage", "Dead dragon"):
            enemy_team = "blue" if attacker.team == "red" else "red"
            targets = [u for u in self.combined if u.team == enemy_team and self._alive(u)]
            if targets:
                atype = attacker.unit_type
                self._log(f"{attacker.team.upper()} {attacker.name}#{attacker.position} ({atype}) применяет массовую атаку.")
                for victim in targets:
                    if not self._roll_hit(attacker):
                        self._log(
                            f"💨 Промах: {attacker.team.upper()} {attacker.name}#{attacker.position} по "
                            f"{victim.team.upper()} {victim.name}#{victim.position}."
                        )
                        continue
                    if self._is_immune_damage(attacker, victim):
                        self._log(
                            f"🛡 Иммунитет к урону '{attacker.attack_type1}' — "
                            f"{victim.team.upper()} {victim.name}#{victim.position}: урон 0."
                        )
                        continue
                    if self._resilience_blocks(attacker, victim):
                        continue
                    before = victim.health
                    dmg = self._apply_damage_with_armor(attacker.damage, victim)
                    victim.health -= dmg
                    after = victim.health
                    self._log(
                        f"{attacker.team.upper()} {attacker.name}#{attacker.position} → "
                        f"{victim.team.upper()} {victim.name}#{victim.position}: {dmg} "
                        f"({before}→{max(0, after)})"
                    )
                    if victim.health <= 0:
                        victim.initiative = 0
                        self._log(f"✖ {victim.team.upper()} {victim.name}#{victim.position} выведен из строя.")
                    else:
                        if atype == "Dead dragon" and attacker.attack_type2 == "poison":
                            acc2 = float(attacker.accuracy2 or 0)
                            roll = self._roll_status(acc2)
                            self._log(
                                f"🧪 Шанс отравления {int(acc2)}% — " + ("успех" if roll else "неудача")
                                + f" по {victim.team.upper()} {victim.name}#{victim.position}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"🛡 Иммунитет к эффекту '{attacker.attack_type2}' — яд НЕ накладывается "
                                        f"на {victim.team.upper()} {victim.name}#{victim.position}."
                                    )
                                else:
                                    victim.poison_turns_left = POISON_TURNS
                                    victim.poison_damage_per_tick = int(attacker.damage2 or 0)
                                    self._log(
                                        f"☠ {attacker.team.upper()} {attacker.name}#{attacker.position} накладывает яд "
                                        f"({victim.poison_damage_per_tick} урона/ход) на "
                                        f"{victim.team.upper()} {victim.name}#{victim.position} на {POISON_TURNS} хода."
                                    )
                return True, "aoe"
            else:
                self._log(
                    f"{attacker.team.upper()} {attacker.name}#{attacker.position} ({attacker.unit_type}) применяет массовую атаку: целей нет."
                )
                return False, "aoe_no_targets"

        # Single-target
        victim = self._unit_by_position(target_pos) if target_pos is not None else None
        if victim is not None and self._alive(victim):
            if not self._roll_hit(attacker):
                self._log(
                    f"💨 Промах: {attacker.team.upper()} {attacker.name}#{attacker.position} по "
                    f"{victim.team.upper()} {victim.name}#{victim.position}."
                )
                return True, "miss"

            if self._is_immune_damage(attacker, victim):
                self._log(
                    f"🛡 Иммунитет к урону '{attacker.attack_type1}' — "
                    f"{victim.team.upper()} {victim.name}#{victim.position}: атака наносит 0."
                )
                return True, "immune_damage"

            if self._resilience_blocks(attacker, victim):
                return True, "resilience_block"

            before = victim.health
            dmg = self._apply_damage_with_armor(attacker.damage, victim)
            victim.health -= dmg
            after = victim.health
            self._log(
                f"{attacker.team.upper()} {attacker.name}#{attacker.position} → "
                f"{victim.team.upper()} {victim.name}#{victim.position}: {dmg} "
                f"({before}→{max(0, after)})"
            )

            if self._alive(victim):
                if (
                    attacker.unit_type == "Death"
                    and attacker.attack_type2 == "poison"
                    and victim.poison_turns_left <= 0
                ):
                    acc2 = float(attacker.accuracy2 or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"🧪 Шанс отравления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim.team.upper()} {victim.name}#{victim.position}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"🛡 Иммунитет к эффекту '{attacker.attack_type2}' — яд НЕ накладывается "
                                f"на {victim.team.upper()} {victim.name}#{victim.position}."
                            )
                        else:
                            victim.poison_turns_left = POISON_TURNS
                            victim.poison_damage_per_tick = int(attacker.damage2 or 0)
                            self._log(
                                f"☠ {attacker.team.upper()} {attacker.name}#{attacker.position} накладывает яд "
                                f"({victim.poison_damage_per_tick} урона/ход) на "
                                f"{victim.team.upper()} {victim.name}#{victim.position} на {POISON_TURNS} хода."
                            )

                if attacker.unit_type == "lord":
                    pos = attacker.position
                    if not self._lord_applied_burn.get(pos, False):
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"🛡 Иммунитет к эффекту '{attacker.attack_type2}' — поджог НЕ накладывается "
                                f"на {victim.team.upper()} {victim.name}#{victim.position}."
                            )
                        else:
                            victim.burn_turns_left = BURN_TURNS
                            victim.burn_damage_per_tick = int(attacker.damage2 or 0)
                            self._lord_applied_burn[pos] = True
                            self._log(
                                f"🔥 {attacker.team.upper()} {attacker.name}#{pos} накладывает поджог "
                                f"({victim.burn_damage_per_tick} урона/ход) на "
                                f"{victim.team.upper()} {victim.name}#{victim.position} на {BURN_TURNS} хода."
                            )

            if victim.health <= 0:
                victim.initiative = 0
                self._log(f"✖ {victim.team.upper()} {victim.name}#{victim.position} выведен из строя.")
            return True, "ok"
        else:
            self._log(
                f"{attacker.team.upper()} {attacker.name}#{attacker.position} бьёт pos{target_pos}: цели нет/мертва/недоступна."
            )
            return False, "dead_or_absent"

    def _check_victory_after_hit(self):
        if not self._team_alive("blue"):
            self.winner = "red";  self._log("🏆 Победа RED!")
        elif not self._team_alive("red"):
            self.winner = "blue"; self._log("🏆 Победа BLUE!")

    def _advance_until_blue_turn(self):
        """
        Автопрогон боя до следующего хода BLUE или до конца битвы.

        Общая идея:
        - Пока обе команды живы, выбирается следующий юнит по инициативе.
        - Если в текущем раунде больше нет юнитов с инициативой, завершается раунд,
          всем живым восстанавливается базовая инициатива, начинается новый раунд.
        - Применяются эффекты начала хода (яд/поджог); они могут убить юнита и завершить бой.
        - Если наступил ход BLUE, сохраняем позицию атакующего и выходим, чтобы агент мог выбрать действие.
        - Если ход RED, он играет автоматически (один или два удара) по правилам ближнего/массового боя.
          После каждого удара проверяется победа. Если бой завершён — выходим.
        Возвращает True, если успешно дошли до хода BLUE; False — если раунд продолжить нельзя (бой завершён).
        """
        while self._team_alive("red") and self._team_alive("blue"):
            # Берём следующего по инициативе живого юнита.
            nxt = self._pop_next()
            if nxt is None:
                # В раунде больше нет активной инициативы: закрываем раунд, восстанавливаем инициативу и начинаем новый.
                self._log(f"— Конец раунда {self.round_no}.")
                self._end_round_restore()
                self.round_no += 1
                self._log(f"— Начало раунда {self.round_no}.")
                continue

            # Эффекты начала хода (tick яда/поджога). Могут убить юнита и завершить бой.
            if not self._apply_start_of_turn_effects(nxt):
                if self.winner is not None:
                    return False
                continue

            # Если наступил ход BLUE — сохраняем текущего атакующего и выходим,
            # чтобы агент принял решение. У Demon возможен второй удар при Twohits=1.
            if nxt.team == "blue":
                self.current_blue_attacker_pos = nxt.position
                self._log(
                    f"Ход BLUE: {nxt.name}#{nxt.position} (иниц {nxt.initiative}). "
                    + ("Ожидание действия (демон: 2 удара)." if (nxt.unit_type == "Demon" and nxt.Twohits == 1) else "Ожидание действия.")
                )
                return True

            # RED ходит автоматически. Тратим инициативу и определяем количество ударов по Twohits.
            nxt.initiative = 0
            strikes = 2 if (nxt.unit_type == "Demon" and getattr(nxt, "Twohits", 0) == 1) else 1

            for hit_i in range(strikes):
                if self.winner is not None:
                    return False

                target_pos = None
                # Ближний бой: выбираем достижимую цель по правилам колонок/рядов
                if nxt.unit_type in ("Воин", "Demon", "lord"):
                    options = self._warrior_allowed_targets(nxt)
                    if options:
                        target_pos = self._pick_lowest_hp(options)
                        prefix = nxt.unit_type
                        self._log(
                            f"RED ход: {nxt.name}#{nxt.position} ({prefix}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt.name}#{nxt.position} ({nxt.unit_type}) не может атаковать: доступных целей нет."
                        )
                # Массовая атака: первый удар — AoE, второй не выполняется
                elif nxt.unit_type in ("Mage", "Dead dragon"):
                    if hit_i == 0:
                        self._log(
                            f"RED ход: {nxt.name}#{nxt.position} ({nxt.unit_type}) выполняет массовую атаку."
                        )
                    else:
                        break
                else:
                    # Прочие типы: одиночная атака по синему юниту с минимальным HP
                    live_blue_positions = self._live_positions_of("blue")
                    target_pos = self._pick_lowest_hp(live_blue_positions) if live_blue_positions else None
                    ut = nxt.unit_type
                    self._log(
                        f"RED ход: {nxt.name}#{nxt.position} ({ut}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes})."
                    )

                # Выполняем удар, если есть цель, либо AoE-тип (Mage/Dead dragon)
                if target_pos is not None or nxt.unit_type in ("Mage", "Dead dragon"):
                    self._attack(nxt, target_pos)
                    # После первого удара демона сбрасываем Twohits в 0
                    if nxt.unit_type == "Demon" and hit_i == 0 and getattr(nxt, "Twohits", 0) == 1:
                        nxt.Twohits = 0
                    self._check_victory_after_hit()
                    if self.winner is not None:
                        return False
        return False

    # ------------------ Observation API ------------------
    def _obs(self):
        vec: List[float] = []
        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = self._unit_by_position(pos)
            hp      = float(max(0, u.health))
            ini     = float(max(0, u.initiative))
            ini_b   = float(u.base_initiative)
            dmg     = float(u.damage)
            dmg2    = float(u.damage2)
            team_v  = 0.0 if u.team == "red" else 1.0
            pos_v   = float(u.position)
            stand_v = 0.0 if u.stand == "ahead" else 1.0

            t_onehot   = _one_hot(u.unit_type, TYPE_LIST)
            imm_mhot   = _multi_hot(u.immunity, ATTACK_TYPES)
            atk1_oh    = _one_hot(u.attack_type1, ATTACK_TYPES) if u.attack_type1 in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            atk2_oh    = _one_hot(u.attack_type2, ATTACK_TYPES) if u.attack_type2 in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            res_mhot   = _multi_hot(u.resilience, ATTACK_TYPES)
            armor_v    = float(u.armor)
            acc_v      = float(u.accuracy)
            acc2_v     = float(u.accuracy2)
            poison_v   = float(u.poison_turns_left)
            burn_v     = float(u.burn_turns_left)

            vec.extend([
                hp, ini, ini_b, dmg, dmg2, team_v, pos_v, stand_v,
                *t_onehot, *imm_mhot, *atk1_oh, *atk2_oh, *res_mhot,
                armor_v, acc_v, acc2_v, poison_v, burn_v,
            ])
        return np.array(vec, dtype=np.float32)

    # ------------------ Gymnasium API ------------------
    def reset(self, *, seed=None, options=None):
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def step(self, action):
        """
        Выполняет один шаг среды для синей стороны (агента).

        Параметры:
            action: индекс действия из Discrete(6). Соответствует выбору цели среди врагов:
                    TARGET_POSITIONS[action] → pos1..pos6 (RED). Для Mage/Dead dragon цель игнорируется (удар по всем).

        Возвращает:
            obs, reward, terminated, truncated, info — как в Gymnasium:
                - obs: текущее наблюдение (np.ndarray)
                - reward: скаляр награды за шаг с учётом shaping
                - terminated: завершён ли эпизод победой/поражением
                - truncated: принудительная остановка (здесь всегда False)
                - info: свободный словарь (здесь пустой)

        Логика шага (высокоуровнево):
            1) Если сейчас не ход BLUE, автоматически прокручиваем бой до хода BLUE или конца боя.
            2) Декодируем action → target_pos. Для Mage/Dead dragon target_pos не используется.
            3) Проверяем, может ли атакующий ударить (жив, имеет инициативу; у Demon возможен второй удар).
            4) Выполняем атаку: промах/иммунитет/стойкость/урон/эффекты (яд/поджог) и фиксация победы, если цель повержена.
            5) Штрафы shaping:
               - penalty_unreachable_warrior — если воин пытается ударить недоступную позицию
               - penalty_invalid_target — если цели нет/мертва (для одиночной атаки)
            6) Если у Demon остался второй удар, возвращаем obs немедленно без смены хода (terminated=False).
            7) Иначе передаём ход (RED до следующего BLUE), считаем базовую награду:
               reward_step — в обычном случае, reward_win/reward_loss — при завершении; добавляем shaping и возвращаем.
        """
        # Предохранитель: эпизод не должен быть завершён к моменту вызова
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        # Если по какой-то причине сейчас не ход BLUE — докручиваем до его хода
        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self.winner is not None:
                base = self.reward_win if self.winner == "blue" else self.reward_loss
                return self._obs(), base, True, False, {}

        # Shaping за неэффективные действия на шаге
        step_shaping = 0.0

        # Декодируем действие агента в позицию цели RED (pos1..pos6)
        target_pos = TARGET_POSITIONS[int(action)]

        # Текущий атакующий BLUE юнит
        attacker = self._unit_by_position(self.current_blue_attacker_pos)
        # Может ли он ударить сейчас (жив, есть инициатива; для Demon разрешён второй удар подряд по Twohits)
        can_strike = (
            attacker is not None
            and self._alive(attacker)
            and (
                attacker.initiative > 0
                or (attacker.unit_type == "Demon" and getattr(attacker, "Twohits", 0) == 0)
            )
        )
        if can_strike:
            was_demon_first_strike = attacker.unit_type == "Demon" and getattr(attacker, "Twohits", 0) == 1
            # Логируем выбранное действие
            self._log(f"BLUE действие: {attacker.name}#{attacker.position} → pos{target_pos}")
            # Тратим инициативу на удар (для Demon второй удар будет доступен при Twohits=0)
            attacker.initiative = 0

            # Ветвь ближнего боя: проверяем достижимые цели по правилам колонок/рядов
            if attacker.unit_type in ("Воин", "Demon", "lord"):
                allowed = self._warrior_allowed_targets(attacker)
                if target_pos not in allowed:
                    self._log(
                        f"BLUE {attacker.name}#{attacker.position} ({attacker.unit_type}) не может достать pos{target_pos}. "
                        f"Доступные цели: {(', '.join('pos'+str(p) for p in allowed)) if allowed else 'нет'}"
                    )
                    step_shaping += self.penalty_unreachable_warrior
                else:
                    hit, reason = self._attack(attacker, target_pos)
                    if not hit and reason == "dead_or_absent":
                        step_shaping += self.penalty_invalid_target
                    self._check_victory_after_hit()
            # Ветвь массовой атаки: цель игнорируется, урон по всем врагам
            elif attacker.unit_type in ("Mage", "Dead dragon"):
                hit, reason = self._attack(attacker, target_pos)
                self._check_victory_after_hit()
            # Прочие типы — одиночная атака без ограничений досягаемости
            else:
                hit, reason = self._attack(attacker, target_pos)
                if not hit and reason == "dead_or_absent":
                    step_shaping += self.penalty_invalid_target
                self._check_victory_after_hit()

            # После первого удара демона флаг Twohits должен стать 0
            if was_demon_first_strike and self.winner is None:
                attacker.Twohits = 0

        # Обработка возможного второго удара у Demon (по Twohits) и перехода хода
        if self.winner is None:
            if attacker is not None and attacker.unit_type == "Demon":
                # Если это был первый удар демона (мы перевели Twohits из 1 в 0), оставляем ход BLUE для второго удара
                if 'was_demon_first_strike' in locals() and was_demon_first_strike:
                    reward, terminated = self.reward_step + step_shaping, False
                    return self._obs(), reward, terminated, False, {}
            # Иначе — передаём ход дальше
            self.current_blue_attacker_pos = None
            self._advance_until_blue_turn()

        # Финальный расчёт базовой награды и признака завершения
        if self.winner is None:
            base = self.reward_step
            terminated = False
        else:
            base = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        reward = base + step_shaping
        return self._obs(), reward, terminated, False, {}


__all__ = [
    "BattleEnv",
    "TYPE_LIST",
    "ATTACK_TYPES",
    "UNITS_RED",
    "UNITS_BLUE",
    "TARGET_POSITIONS",
]


def _pos_row_label(pos):
    if pos in (1, 2, 3, 7, 8, 9):
        return "передний ряд"
    return "задний ряд"


def _pos_col_label(pos):
    col = (pos - 1) % 3
    return {0: "левый", 1: "центр", 2: "правый"}.get(col, str(col))


def print_action_space_info():
    print("\n=== Доступные действия агента (Discrete(6)) ===")
    print("Действие i соответствует выбору цели posX среди врагов (RED: pos1..pos6).")
    for i, pos in enumerate(TARGET_POSITIONS):
        print(
            f"  {i}: pos{pos} — RED, {_pos_row_label(pos)}, столбец {_pos_col_label(pos)}"
        )
    print("Примечания:")
    print("- 'Воин'/'Demon'/'lord': ближний бой — цель должна быть достижима по правилам колонок и рядов.")
    print("- 'Mage'/'Dead dragon': массовая атака — выбор posX игнорируется, бьёт по всем целям противника.")
    print(
        "- Если цель недоступна/мертва: действие считается неэффективным и может давать штраф shaping."
    )


def _build_slot_feature_names():
    base = [
        "hp",
        "ini",
        "ini_base",
        "dmg",
        "dmg2",
        "team",  # 0=red, 1=blue
        "pos",
        "stand",  # 0=ahead, 1=behind
    ]
    type_oh = [f"type[{t}]" for t in TYPE_LIST]
    imm = [f"imm[{a}]" for a in ATTACK_TYPES]
    atk1 = [f"atk1[{a}]" for a in ATTACK_TYPES]
    atk2 = [f"atk2[{a}]" for a in ATTACK_TYPES]
    res = [f"res[{a}]" for a in ATTACK_TYPES]
    tail = ["armor", "accuracy", "accuracy2", "poison_left", "burn_left"]
    names = base + type_oh + imm + atk1 + atk2 + res + tail
    return names


def build_all_feature_names():
    per_slot = _build_slot_feature_names()
    all_names: List[str] = []
    for pos in RED_POSITIONS + BLUE_POSITIONS:
        pref = f"pos{pos}."
        all_names.extend([pref + n for n in per_slot])
    return all_names


def print_observation_space_info(env, obs):
    names = build_all_feature_names()
    print("\n=== Observation space ===")
    print(f"Space: {env.observation_space}")
    print("Структура: 12 позиций × 45 признаков = 540 значений.")
    print("Индексация признаков (i → имя):")
    for i, nm in enumerate(names):
        print(f"  {i:3d}: {nm}")
    # Полные границы пространства
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
    try:
        print("\nГраницы low:")
        print(env.observation_space.low)
        print("Границы high:")
        print(env.observation_space.high)
    except Exception:
        pass
    # Текущее наблюдение
    print("\nТекущее наблюдение obs (после reset):")
    print(obs)


def _describe_unit(u):
    alive = u.health > 0
    hp_str = f"{max(0, u.health)}/{u.maxhealth}"
    ini_str = f"{u.initiative}/{u.base_initiative}"
    row = _pos_row_label(u.position)  # передний/задний ряд
    col = _pos_col_label(u.position)  # левый/центр/правый
    status_parts: List[str] = []
    if u.poison_turns_left > 0:
        status_parts.append(f"☠ poison {u.poison_damage_per_tick}/ход · {u.poison_turns_left} хода")
    if u.burn_turns_left > 0:
        status_parts.append(f"🔥 burn {u.burn_damage_per_tick}/ход · {u.burn_turns_left} хода")
    status = ("; ".join(status_parts)) if status_parts else "—"
    type_tag = u.unit_type
    atk1 = u.attack_type1
    atk2 = u.attack_type2
    acc = int(u.accuracy or 0)
    acc2 = int(u.accuracy2 or 0)
    arm = int(u.armor or 0)
    big = ", BIG" if u.big else ""
    life = "ALIVE" if alive else "DEAD"
    return (
        f"pos{u.position:>2} {u.team.upper():>4} {u.name} ({type_tag}{big}) | "
        f"HP {hp_str} | INI {ini_str} | ARM {arm} | ATK {atk1}{('/'+atk2) if atk2 else ''} | "
        f"ACC {acc}/{acc2} | {row}, {col} | {life} | статус: {status}"
    )


def render_battle_state(env):
    print("RED команда:")
    for pos in RED_POSITIONS:
        u = env._unit_by_position(pos)
        if u is not None:
            print("  " + _describe_unit(u))
    print("BLUE команда:")
    for pos in BLUE_POSITIONS:
        u = env._unit_by_position(pos)
        if u is not None:
            print("  " + _describe_unit(u))


def choose_action_min_hp(env):
    attacker_pos = env.current_blue_attacker_pos
    if attacker_pos is None:
        return 0
    attacker = env._unit_by_position(attacker_pos)
    if attacker is None:
        return 0
    if attacker.unit_type in ("Воин", "Demon", "lord"):
        allowed = env._warrior_allowed_targets(attacker)
    else:
        allowed = env._live_positions_of("red")
    if not allowed:
        allowed = env._live_positions_of("red")
    if not allowed:
        return 0
    # Choose min HP among allowed
    best_pos = None
    best_hp = None
    for p in allowed:
        uu = env._unit_by_position(p)
        if uu is None or uu.health <= 0:
            continue
        if best_hp is None or uu.health < best_hp:
            best_hp = uu.health
            best_pos = p
    if best_pos is None:
        best_pos = allowed[0]
    try:
        return TARGET_POSITIONS.index(best_pos)
    except ValueError:
        return 0


def run_demo(num_actions, seed = 42):
    env = BattleEnv(log_enabled=True)
    env.seed(seed)
    env.reset()
    actions_done = 0
    while actions_done < num_actions and env.winner is None:
        act = choose_action_min_hp(env)
        _, _, terminated, _, _ = env.step(act)
        actions_done += 1
        # Print turn log and current state
        events = env.pop_pretty_events()
        if events:
            print("\n— События после действия", actions_done, "—")
            for line in events:
                print(line)
        print(f"\nСостояние после {actions_done} действия(й):")
        render_battle_state(env)
        if terminated:
            break


if __name__ == "__main__":
    # Демонстрация среды: вывод состояния после 1-го и 5-го действия агента
    print("=== Демонстрация: после 1 действия агента ===")
    run_demo(num_actions=1, seed=42)
    print("\n=== Демонстрация: после 5 действий агента ===")
    run_demo(num_actions=20, seed=42)

