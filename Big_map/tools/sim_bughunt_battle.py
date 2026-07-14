"""Bug-hunt симуляция произвольного боя BattleEnv.

По умолчанию серия сбалансирована по управлению: в чётных запусках исходным BLUE
управляет обученный campaign-agent, а RED — скриптовый бот; в нечётных команды
зеркалируются, поэтому агент управляет исходным RED, а BLUE получает того же бота.
После каждого шага проверяются инварианты; расхождения печатаются в отчёте.

Состав команд задаётся строкой "позиция:Имя,позиция:Имя,..." (имена — из DATA).
RED занимает позиции 1-6 (1-3 передний ряд), BLUE — 7-12 (7-9 передний ряд).

Запуск:
    python tools/sim_bughunt_battle.py --battles 300
    python tools/sim_bughunt_battle.py --battles 1 --seed 7 --dump-log
    python tools/sim_bughunt_battle.py --red "1:Астерот,3:Владыка,5:Инкуб" ^
        --blue "7:Король гномов,8:Мастер клинка,9:Защитник Веры,10:Патриарх,12:Имперский ассасин"
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import battle_env as be
from battle_env import BattleEnv, _apply_hero_levelup_bonuses
from console_encoding import setup_utf8_console
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

setup_utf8_console()


def _entry(name: str) -> dict:
    for e in DATA:
        if isinstance(e, dict) and str(e.get("кто", "")).strip() == name:
            return e
    raise KeyError(name)


DEFAULT_RED_SPEC = "3:Грифон,5:Мудрец"
DEFAULT_BLUE_SPEC = "8:Дракон рока,12:Патриарх"


def analyze_battle_log(log_lines: List[str]) -> Tuple[Counter, List[str]]:
    """Aggregate exercised mechanics and observable D2/invariant discrepancies."""
    events: Counter = Counter()
    bugs: List[str] = []
    for line in log_lines:
        if "призывает" in line or ": призван " in line:
            events["призывы"] += 1
        if "Ведьма меняет характеристики" in line:
            events["превращения в беса"] += 1
        if "Воскрешение Патриарха" in line:
            events["воскрешения Патриарха"] += 1
        if "накладывает поджог" in line:
            events["поджоги"] += 1
        if "накладывает воду" in line:
            events["эффекты воды"] += 1
        if "Яд:" in line or "накладывает яд" in line:
            events["отравления"] += 1
        if "Паралич:" in line or "Долгий паралич:" in line:
            events["параличи"] += 1
        if "снижения брони" in line or "armor shred" in line.lower():
            events["проверки разрушения брони"] += 1
        if "гибнет вместе с призывателем" in line:
            events["гибель связанных призывов"] += 1

        stripped = line.strip()
        if (
            (stripped.startswith("{") and stripped.endswith("}"))
            or " ROW CHECK " in line
            or "ahead_unit={" in line
        ):
            bugs.append(f"В боевой лог попал отладочный словарь/служебная строка: {line}")

    return events, bugs


def _mirror_team(units: List[dict], target_team: str) -> List[dict]:
    """Mirror a starting team between RED 1..6 and BLUE 7..12 positions."""
    mirrored = deepcopy(units)
    for unit in mirrored:
        position = int(unit.get("position", 0) or 0)
        if target_team == "blue":
            position = position + 6 if position in range(1, 7) else position
        else:
            position = position - 6 if position in range(7, 13) else position
        unit["team"] = target_team
        unit["position"] = position
        unit["stand"] = "ahead" if position in (*be.RED_FRONT_POSITIONS, *be.BLUE_FRONT_POSITIONS) else "behind"
    return mirrored


def balanced_controlled_team(battle_index: int) -> str:
    """Alternate the externally controlled original side for a 50/50 even series."""
    return "blue" if int(battle_index) % 2 == 0 else "red"


class CampaignBattleAgent:
    """Use the latest trained campaign policy for standalone BattleEnv decisions."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vecnormalize_path: Optional[str] = None,
    ) -> None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from sb3_contrib.ppo_mask import MaskablePPO

        from bestmodeltest import make_env

        if model_path:
            resolved_model = Path(model_path).resolve()
        else:
            candidates = list(PROJECT_ROOT.glob("campaign_models/*/best/best_model.zip"))
            if not candidates:
                raise FileNotFoundError("No campaign_models/*/best/best_model.zip found")
            resolved_model = max(candidates, key=lambda path: path.stat().st_mtime)

        if vecnormalize_path:
            resolved_vecnormalize = Path(vecnormalize_path).resolve()
        else:
            candidates = (
                resolved_model.with_name("best_vecnormalize.pkl"),
                resolved_model.parent.parent / "vecnormalize.pkl",
            )
            resolved_vecnormalize = next((path for path in candidates if path.is_file()), None)
            if resolved_vecnormalize is None:
                raise FileNotFoundError(f"VecNormalize stats not found for {resolved_model}")

        normalization_env = DummyVecEnv([make_env])
        self.vecnormalize = VecNormalize.load(str(resolved_vecnormalize), normalization_env)
        self.vecnormalize.training = False
        self.vecnormalize.norm_reward = False
        self.model = MaskablePPO.load(str(resolved_model))

        wrapped_campaign = make_env()
        current = wrapped_campaign
        while hasattr(current, "env"):
            current = current.env
        self.campaign_env = current
        self.campaign_env.reset(seed=0)
        self.model_path = resolved_model
        self.vecnormalize_path = resolved_vecnormalize

    def action_for(self, battle: BattleEnv) -> int:
        campaign = self.campaign_env
        campaign.mode = campaign.MODE_BATTLE
        campaign.battle_env = battle
        obs = campaign._build_obs(battle_obs=battle._obs())
        normalized_obs = self.vecnormalize.normalize_obs(
            np.asarray(obs, dtype=np.float32).reshape(1, -1)
        )
        action_mask = campaign.compute_action_mask().reshape(1, -1)
        action, _ = self.model.predict(
            normalized_obs,
            deterministic=True,
            action_masks=action_mask,
        )
        return int(np.asarray(action).reshape(-1)[0])


def parse_team_spec(spec: str) -> Dict[int, Tuple[str, int]]:
    """Разбирает "поз:Имя" или "поз:Имя:уровень" (уровень героя, по умолчанию 1)."""
    placement: Dict[int, Tuple[str, int]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pos_str, _, rest = chunk.partition(":")
        name, _, level_str = rest.partition(":")
        level = int(level_str) if level_str.strip() else 1
        placement[int(pos_str)] = (name.strip(), level)
    return placement


def build_team(team: str, spec: str, positions: range) -> List[dict]:
    placement = parse_team_spec(spec)
    units: List[dict] = []
    for pos in positions:
        name, level = placement.get(pos, ("", 1))
        if not name:
            units.append(placeholder_unit(team, pos))
            continue
        unit = map_unit_to_battle(_entry(name), team, pos)
        for _ in range(max(0, level - 1)):
            _apply_hero_levelup_bonuses(unit)
        if level > 1:
            unit["Level"] = level
        units.append(unit)
    return units


def _is_placeholder(u: dict) -> bool:
    return not str(u.get("name", "")).strip() or str(u.get("name", "")).strip() == "пусто"


def _alive(u: dict) -> bool:
    try:
        return int(u.get("health", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


class BattleAudit:
    def __init__(self, seed: int):
        self.seed = seed
        self.anomalies: List[str] = []
        # (team, position) -> имя текущего живого/последнего умершего юнита
        self.deaths: List[dict] = []  # {step, team, position, name, exp_kill}
        self.overwritten_dead: List[dict] = []
        self.transforms: List[dict] = []
        self._last_state: Dict[Tuple[str, int], dict] = {}

    def flag(self, step: int, msg: str) -> None:
        self.anomalies.append(f"[seed={self.seed} step={step}] {msg}")

    def snapshot(self, env: BattleEnv, step: int) -> None:
        state: Dict[Tuple[str, int], dict] = {}
        for u in env.combined:
            if not isinstance(u, dict) or _is_placeholder(u):
                continue
            key = (str(u.get("team", "")), int(u.get("position", 0) or 0))
            state[key] = {
                "name": str(u.get("name", "")),
                "alive": _alive(u),
                "exp_kill": float(u.get("exp_kill", 0) or 0),
                "health": int(u.get("health", 0) or 0),
                "transformed": bool(u.get("transformed", 0))
                or bool(u.get("basestats"))
                or bool(u.get("wight_form_name")),
                "summoned": u.get("Summoned") is not None,
            }

        for key, cur in state.items():
            prev = self._last_state.get(key)
            if prev is None:
                # Новый юнит в слоте (призыв). Если раньше тут лежал труп — он затёрт.
                continue
            if prev["name"] == cur["name"]:
                if prev["alive"] and not cur["alive"]:
                    self.deaths.append(
                        {
                            "step": step,
                            "team": key[0],
                            "position": key[1],
                            "name": cur["name"],
                            "exp_kill": cur["exp_kill"],
                        }
                    )
            else:
                # Имя в слоте сменилось: превращение или призыв поверх старого юнита
                if prev["transformed"] or cur["transformed"]:
                    self.transforms.append(
                        {
                            "step": step,
                            "team": key[0],
                            "position": key[1],
                            "old_name": prev["name"],
                            "new_name": cur["name"],
                        }
                    )
                elif prev["alive"]:
                    if cur["summoned"]:
                        # Юнит погиб и был затёрт призывом внутри одного шага:
                        # фиксируем смерть и затирание, а не нелегальную подмену.
                        self.deaths.append(
                            {
                                "step": step,
                                "team": key[0],
                                "position": key[1],
                                "name": prev["name"],
                                "exp_kill": prev["exp_kill"],
                            }
                        )
                        self.overwritten_dead.append(
                            {
                                "step": step,
                                "team": key[0],
                                "position": key[1],
                                "old_name": prev["name"],
                                "old_exp_kill": prev["exp_kill"],
                                "new_name": cur["name"],
                            }
                        )
                    else:
                        self.flag(
                            step,
                            f"ЖИВОЙ юнит затёрт: {key} {prev['name']} -> {cur['name']}",
                        )
                else:
                    self.overwritten_dead.append(
                        {
                            "step": step,
                            "team": key[0],
                            "position": key[1],
                            "old_name": prev["name"],
                            "old_exp_kill": prev["exp_kill"],
                            "new_name": cur["name"],
                        }
                    )
        # Юниты, исчезнувшие из слота (без замены): легально только через побег
        escaped_names = {
            (str(u.get("team", "")), str(u.get("name", "")))
            for u in getattr(env, "escaped_units", []) or []
            if isinstance(u, dict)
        }
        for key, prev in self._last_state.items():
            if key not in state:
                if (key[0], prev["name"]) in escaped_names:
                    continue
                self.flag(step, f"Юнит исчез из слота {key}: {prev['name']}")
        self._last_state = state

    def check_invariants(self, env: BattleEnv, step: int, obs: Optional[np.ndarray]) -> None:
        alive_positions: List[int] = []
        for u in env.combined:
            if not isinstance(u, dict) or _is_placeholder(u):
                continue
            name = str(u.get("name", ""))
            team = str(u.get("team", ""))
            pos = int(u.get("position", 0) or 0)
            hp = int(u.get("health", 0) or 0)
            max_hp = int(u.get("max_health", 0) or 0)

            if hp > max_hp:
                self.flag(step, f"{team} {name}#{pos}: HP {hp} > max {max_hp}")
            if hp < 0:
                self.flag(step, f"{team} {name}#{pos}: отрицательный HP {hp}")
            if team == "red" and pos not in range(1, 7):
                self.flag(step, f"RED юнит {name} на чужой позиции {pos}")
            if team == "blue" and pos not in range(7, 13):
                self.flag(step, f"BLUE юнит {name} на чужой позиции {pos}")

            for eff, cap in (
                ("poison_turns_left", be.POISON_TURNS),
                ("burn_turns_left", be.BURN_TURNS),
                ("uran_turns_left", be.URAN_TURNS),
            ):
                val = int(u.get(eff, 0) or 0)
                if val < 0 or val > cap:
                    self.flag(step, f"{team} {name}#{pos}: {eff}={val} вне [0..{cap}]")

            if _alive(u):
                alive_positions.append(pos)
                if bool(u.get("big", False)):
                    if pos in (4, 5, 6, 10, 11, 12):
                        self.flag(step, f"Большой юнит {name} в заднем ряду pos{pos}")
                    partner = pos + 3
                    partner_units = [
                        v
                        for v in env.combined
                        if isinstance(v, dict)
                        and not _is_placeholder(v)
                        and int(v.get("position", 0) or 0) == partner
                        and _alive(v)
                    ]
                    if partner_units:
                        self.flag(
                            step,
                            f"Большой {name}#{pos}: партнёрская клетка pos{partner} "
                            f"занята живым {partner_units[0].get('name')}",
                        )

        dup = [p for p, c in Counter(alive_positions).items() if c > 1]
        if dup:
            self.flag(step, f"Две живые единицы на одной позиции: {dup}")

        if obs is not None:
            arr = np.asarray(obs, dtype=np.float32)
            if np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6):
                bad_idx = np.where((arr < -1e-6) | (arr > 1.0 + 1e-6))[0][:5]
                vals = [f"{i}:{arr[i]:.3f}" for i in bad_idx]
                self.flag(step, f"Наблюдение вне [0,1]: {', '.join(vals)}")

    def finalize(self, env: BattleEnv, step: int, step_cap_hit: bool) -> dict:
        result = {
            "seed": self.seed,
            "winner": env.winner,
            "steps": step,
            "rounds": env.round_no,
            "step_cap_hit": step_cap_hit,
            "anomalies": self.anomalies,
            "deaths": self.deaths,
            "overwritten_dead": self.overwritten_dead,
            "transforms": self.transforms,
            "battle_exp": float(getattr(env, "last_battle_exp", 0.0) or 0.0),
        }
        if step_cap_hit:
            self.flag(step, "Достигнут лимит шагов — бой не завершился (возможно, вечный бой)")
            result["anomalies"] = self.anomalies
            return result

        if env.winner in ("red", "blue"):
            losing = "red" if env.winner == "blue" else "blue"
            expected = 0.0
            counted: set = set()
            for d in self.deaths:
                if d["team"] != losing:
                    continue
                key = (d["team"], d["position"], d["name"])
                # Один и тот же юнит мог умереть, быть воскрешён и умереть снова —
                # опыт должен считаться один раз (как одна убитая единица в конце боя)
                if key in counted:
                    continue
                counted.add(key)
                expected += float(d["exp_kill"])
            # Затёртые призывом трупы из проигравшей команды всё равно были убиты
            actual = float(getattr(env, "last_battle_exp", 0.0) or 0.0)
            if env.winner == "blue" and abs(expected - actual) > 1e-6:
                self.flag(
                    step,
                    f"ОПЫТ: начислено {actual:g}, а убито на {expected:g} "
                    f"(потеряно {expected - actual:g})",
                )
            result["expected_exp"] = expected
            result["anomalies"] = self.anomalies
        return result


def run_battle(
    seed: int,
    *,
    step_cap: int = 3000,
    dump_log: bool = False,
    red_spec: str = DEFAULT_RED_SPEC,
    blue_spec: str = DEFAULT_BLUE_SPEC,
    controlled_team: str = "blue",
    controller: str = "scripted",
    agent: Optional[CampaignBattleAgent] = None,
    log_dir: Optional[Path] = None,
) -> dict:
    rng = random.Random(seed)
    random.seed(seed)  # summon_actions и часть battle_env используют модульный random

    env = BattleEnv(log_enabled=True)
    env.rng = random.Random(seed + 1)

    original_red = build_team("red", red_spec, range(1, 7))
    original_blue = build_team("blue", blue_spec, range(7, 13))
    mirrored = controlled_team == "red"
    if mirrored:
        red = _mirror_team(original_blue, "red")
        blue = _mirror_team(original_red, "blue")
    else:
        red = original_red
        blue = original_blue
    audit = BattleAudit(seed)

    log_lines: List[str] = []
    env._init_with_custom_teams(red, blue)
    log_lines.extend(env.pop_pretty_events())
    audit.snapshot(env, 0)
    audit.check_invariants(env, 0, None)

    step = 0
    step_cap_hit = False
    while env.winner is None:
        step += 1
        if step > step_cap:
            step_cap_hit = True
            break
        if controller == "agent":
            if agent is None:
                raise ValueError("controller='agent' requires CampaignBattleAgent")
            action = agent.action_for(env)
        elif controller == "scripted":
            action = env.scripted_action_for_current_blue()
        elif controller == "random":
            mask = env.compute_action_mask()
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                audit.flag(step, "Пустая маска действий при незавершённом бое")
                break
            choices = [a for a in valid.tolist() if a != be.RUN_AWAY_ACTION_INDEX]
            if not choices:
                choices = valid.tolist()
            action = int(rng.choice(choices))
        else:
            raise ValueError(f"Unknown controller: {controller}")
        obs, reward, terminated, truncated, info = env.step(action)
        log_lines.extend(env.pop_pretty_events())
        audit.snapshot(env, step)
        audit.check_invariants(env, step, obs)
        if terminated or truncated:
            break

    result = audit.finalize(env, step, step_cap_hit)
    internal_winner = result.get("winner")
    result["internal_winner"] = internal_winner
    if mirrored and internal_winner in ("red", "blue"):
        result["winner"] = "blue" if internal_winner == "red" else "red"
    result["controlled_team"] = controlled_team
    result["controller"] = controller
    result["controllers"] = {
        controlled_team: controller,
        "red" if controlled_team == "blue" else "blue": "scripted",
    }
    result["mirrored"] = mirrored
    event_counts, mechanic_bugs = analyze_battle_log(log_lines)
    result["event_counts"] = dict(event_counts)
    result["mechanic_bugs"] = mechanic_bugs
    if dump_log or result["anomalies"] or mechanic_bugs:
        out_dir = Path(log_dir) if log_dir is not None else PROJECT_ROOT / "outputs" / "bughunt"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"battle_seed{seed}_{controlled_team}_{controller}.log"
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        result["log_path"] = str(log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Начальный сид (сиды идут подряд).")
    parser.add_argument("--step-cap", type=int, default=3000)
    parser.add_argument("--dump-log", action="store_true", help="Сохранять лог каждого боя.")
    parser.add_argument("--red", default=DEFAULT_RED_SPEC, help="Состав RED: 'поз:Имя,...' (1-6).")
    parser.add_argument("--blue", default=DEFAULT_BLUE_SPEC, help="Состав BLUE: 'поз:Имя,...' (7-12).")
    parser.add_argument(
        "--control-mode",
        choices=("balanced", "agent", "scripted", "random"),
        default="balanced",
        help=(
            "balanced: попеременно BLUE-agent/RED-bot и RED-agent/BLUE-bot; "
            "остальные режимы управляют стороной из --controlled-team"
        ),
    )
    parser.add_argument("--controlled-team", choices=("blue", "red"), default="blue")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vecnormalize-path", default=None)
    args = parser.parse_args()

    needs_agent = args.control_mode in ("balanced", "agent")
    agent = (
        CampaignBattleAgent(args.model_path, args.vecnormalize_path)
        if needs_agent
        else None
    )
    if agent is not None:
        print(f"Agent model: {agent.model_path}")
        print(f"VecNormalize: {agent.vecnormalize_path}")

    wins = Counter()
    total_anomalies: List[str] = []
    overwrite_count = 0
    transform_count = 0
    exp_stats: List[Tuple[float, float]] = []
    lengths: List[int] = []
    controller_counts: Counter = Counter()

    for i in range(args.battles):
        seed = args.seed + i
        if args.control_mode == "balanced":
            controlled_team = balanced_controlled_team(i)
            controller = "agent"
        else:
            controlled_team = args.controlled_team
            controller = args.control_mode
        res = run_battle(
            seed,
            step_cap=args.step_cap,
            dump_log=args.dump_log,
            red_spec=args.red,
            blue_spec=args.blue,
            controlled_team=controlled_team,
            controller=controller,
            agent=agent,
        )
        wins[res["winner"] or "stall"] += 1
        for team, active_controller in res["controllers"].items():
            controller_counts[(team, active_controller)] += 1
        lengths.append(res["steps"])
        total_anomalies.extend(res["anomalies"])
        overwrite_count += len(res["overwritten_dead"])
        transform_count += len(res.get("transforms", []))
        if "expected_exp" in res and res["internal_winner"] == "blue":
            exp_stats.append((res["expected_exp"], res["battle_exp"]))
        for od in res["overwritten_dead"]:
            total_anomalies.append(
                f"[seed={seed} step={od['step']}] Труп затёрт призывом: "
                f"{od['team']} pos{od['position']} {od['old_name']} "
                f"(exp_kill={od['old_exp_kill']:g}) -> {od['new_name']}"
            )

    print(f"Боёв: {args.battles}, победы: {dict(wins)}")
    print(
        "Управление: "
        + ", ".join(
            f"{team.upper()} {controller}={count}"
            for (team, controller), count in sorted(controller_counts.items())
        )
    )
    print(f"Длина боя (шаги BLUE): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    if exp_stats:
        mismatched = [(e, a) for e, a in exp_stats if abs(e - a) > 1e-6]
        print(
            f"Побед BLUE с проверкой опыта: {len(exp_stats)}, расхождений: {len(mismatched)}"
        )
        if mismatched:
            lost = sum(e - a for e, a in mismatched)
            print(f"  Суммарно потеряно опыта: {lost:g}")
    print(f"Затираний трупов призывом: {overwrite_count}")
    print(f"Превращений юнитов: {transform_count}")
    print(f"Всего аномалий: {len(total_anomalies)}")
    shown = 0
    seen_kinds: Counter = Counter()
    for a in total_anomalies:
        kind = a.split("]", 1)[-1][:60]
        seen_kinds[kind] += 1
    print("\nТоп типов аномалий:")
    for kind, cnt in seen_kinds.most_common(15):
        print(f"  {cnt:5d}x {kind}")
    print("\nПримеры аномалий:")
    for a in total_anomalies[:30]:
        print(f"  {a}")
        shown += 1


if __name__ == "__main__":
    main()
