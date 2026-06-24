from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE
from train_campaign import REWARD_CONFIG


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.compute_action_mask()


class TerminalSnapshotWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            info["campaign_state_summary"] = self.env.get_state_summary()
            info["built_buildings"] = list(self.env.get_built_building_names())
            info["hero_visual"] = self.env.get_travel_hero_visual_info()
            blue_units = self.env._get_blue_state() if hasattr(self.env, "_get_blue_state") else []
            info["blue_team_final"] = [
                {
                    "name": str(unit.get("name", "") or ""),
                    "position": int(unit.get("position", 0) or 0),
                    "level": int(unit.get("Level", unit.get("level", 0)) or 0),
                    "hp": float(unit.get("health", unit.get("hp", 0.0)) or 0.0),
                    "max_hp": float(unit.get("maxhealth", unit.get("max_hp", 0.0)) or 0.0),
                }
                for unit in blue_units
                if isinstance(unit, dict) and str(unit.get("name", "") or "")
            ]
        return obs, reward, terminated, truncated, info


def make_env():
    def _factory():
        env = CampaignEnv(
            grid_size=DEFAULT_GRID_SIZE,
            **REWARD_CONFIG,
            persist_blue_hp=True,
            log_enabled=False,
            detailed_step_info=True,
            freeze_dynamic_action_layout=True,
            scripted_capital_bot_enabled=False,
            empire_territory_enabled=True,
            campaign_objective="dragon",
            max_grid_steps=1800,
            Realcapital=2,
        )
        return Monitor(ActionMasker(TerminalSnapshotWrapper(env), mask_fn))

    return _factory


def new_tracker() -> dict[str, Any]:
    return {
        "steps": 0,
        "grid_steps": 0,
        "battle_steps": 0,
        "battle_starts": 0,
        "battle_wins": 0,
        "battle_defeats": 0,
        "battle_timeouts": 0,
        "dragon_starts": 0,
        "first_dragon_step": None,
        "spells": 0,
        "dragon_targeted_spells": 0,
        "spell_kinds": Counter(),
        "spell_keys": Counter(),
        "summoned_units": 0,
        "buildings": 0,
        "build_names": Counter(),
        "chests": 0,
        "ruins": 0,
        "hires": 0,
        "unit_swaps": 0,
        "items_equipped": 0,
        "items_used": 0,
        "castle_heals": 0,
        "castle_healed_hp": 0.0,
        "rests": 0,
        "unit_upgrades": 0,
        "enemy_battle_starts": Counter(),
        "enemy_battle_wins": Counter(),
    }


def enemy_id_from_info(info: dict[str, Any]) -> int | None:
    raw = info.get("enemy_id")
    if raw is None:
        raw = info.get("target_enemy_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def update_tracker(tracker: dict[str, Any], info: dict[str, Any]) -> None:
    tracker["steps"] += 1
    mode = str(info.get("mode", "") or "")
    if mode == "battle":
        tracker["battle_steps"] += 1
    else:
        tracker["grid_steps"] += 1

    enemy_id = enemy_id_from_info(info)
    if info.get("battle_triggered"):
        tracker["battle_starts"] += 1
        if enemy_id is not None:
            tracker["enemy_battle_starts"][str(enemy_id)] += 1
        if enemy_id == 31:
            tracker["dragon_starts"] += 1
            if tracker["first_dragon_step"] is None:
                tracker["first_dragon_step"] = tracker["steps"]

    battle_result = str(info.get("battle_result", "") or "")
    if battle_result == "victory":
        tracker["battle_wins"] += 1
        if enemy_id is not None:
            tracker["enemy_battle_wins"][str(enemy_id)] += 1
    elif battle_result == "defeat":
        tracker["battle_defeats"] += 1
    elif battle_result == "timeout":
        tracker["battle_timeouts"] += 1

    if info.get("spell_cast_action") and info.get("spell_cast_executed"):
        tracker["spells"] += 1
        spell_kind = str(info.get("spell_kind", "") or "unknown")
        spell_key = str(info.get("spell_key", "") or "unknown")
        tracker["spell_kinds"][spell_kind] += 1
        tracker["spell_keys"][spell_key] += 1
        if enemy_id == 31:
            tracker["dragon_targeted_spells"] += 1
        if spell_kind == "summon_battle":
            try:
                count = int(info.get("spell_units_affected", 1) or 1)
            except (TypeError, ValueError):
                count = 1
            tracker["summoned_units"] += max(1, count)

    if info.get("build_action") and info.get("built"):
        tracker["buildings"] += 1
        tracker["build_names"][str(info.get("build_name", "") or "unknown")] += 1
    tracker["chests"] += int(info.get("collected_chests_count", 0) or 0)
    tracker["ruins"] += int(bool(info.get("ruin_reward_applied")))
    hired = int(info.get("hired_units_count", 0) or 0)
    tracker["hires"] += hired if hired > 0 else int(bool(info.get("hired")))
    tracker["unit_swaps"] += int(
        bool(info.get("unit_swap_action")) and bool(info.get("unit_swap_applied"))
    )
    tracker["items_equipped"] += int(
        bool(info.get("equip_battle_item_action")) and bool(info.get("equip_battle_item_applied"))
    )
    tracker["items_used"] += int(
        bool(info.get("battle_hero_item_action"))
        and bool(info.get("battle_hero_item_applied"))
        and (
            bool(info.get("battle_hero_item_charge_spent"))
            or bool(info.get("battle_hero_item_consumed"))
        )
    )
    if info.get("castle_heal_action") and float(info.get("healed_amount", 0.0) or 0.0) > 0:
        tracker["castle_heals"] += 1
        tracker["castle_healed_hp"] += float(info.get("healed_amount", 0.0) or 0.0)
    tracker["rests"] += int(bool(info.get("rest_action")))
    tracker["unit_upgrades"] += int(info.get("unit_upgrades", 0) or 0)


def finish_record(tracker: dict[str, Any], info: dict[str, Any], reward: float) -> dict[str, Any]:
    state = dict(info.get("campaign_state_summary", {}) or {})
    episode = dict(info.get("episode", {}) or {})
    record = {
        key: value
        for key, value in tracker.items()
        if not isinstance(value, Counter)
    }
    record.update(
        {
            "result": str(info.get("campaign_result", "unknown") or "unknown"),
            "victory_reason": str(info.get("campaign_victory_reason", "") or ""),
            "is_dragon_win": (
                str(info.get("campaign_result", "") or "") == "victory"
                and str(info.get("campaign_victory_reason", "") or "") == "green_dragon_defeated"
            ),
            "episode_reward": float(episode.get("r", reward) or reward),
            "episode_length": int(episode.get("l", tracker["steps"]) or tracker["steps"]),
            "turns": int(state.get("turns", info.get("turns", 0)) or 0),
            "moves_left": float(state.get("moves", info.get("moves", 0.0)) or 0.0),
            "gold_left": float(state.get("gold", info.get("gold", 0.0)) or 0.0),
            "enemies_defeated": len(state.get("enemies_defeated", []) or []),
            "visited_cells": int(state.get("visited_cells", 0) or 0),
            "learned_spells_count": int(state.get("learned_spells_count", 0) or 0),
            "learned_spells": list(state.get("learned_spells", []) or []),
            "built_buildings": list(info.get("built_buildings", []) or []),
            "hero_visual": info.get("hero_visual"),
            "blue_team_final": list(info.get("blue_team_final", []) or []),
            "blue_alive_ratio_after_dragon": float(info.get("blue_alive_ratio", 0.0) or 0.0),
            "blue_hp_ratio_after_dragon": float(info.get("blue_hp_ratio", 0.0) or 0.0),
            "equipped_hero_items": list(state.get("equipped_hero_items", []) or []),
            "battle_items_equipped_total": int(state.get("battle_items_equipped_total", 0) or 0),
            "battle_items_used_total": int(state.get("battle_items_used_total", 0) or 0),
            "spell_kinds": dict(tracker["spell_kinds"]),
            "spell_keys": dict(tracker["spell_keys"]),
            "build_names": dict(tracker["build_names"]),
            "enemy_battle_starts": dict(tracker["enemy_battle_starts"]),
            "enemy_battle_wins": dict(tracker["enemy_battle_wins"]),
        }
    )
    return record


NUMERIC_FIELDS = (
    "episode_length",
    "episode_reward",
    "turns",
    "gold_left",
    "enemies_defeated",
    "visited_cells",
    "battle_starts",
    "battle_wins",
    "spells",
    "summoned_units",
    "buildings",
    "chests",
    "ruins",
    "hires",
    "unit_swaps",
    "items_equipped",
    "items_used",
    "castle_heals",
    "unit_upgrades",
    "learned_spells_count",
    "first_dragon_step",
    "blue_alive_ratio_after_dragon",
    "blue_hp_ratio_after_dragon",
)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(records)}
    for field in NUMERIC_FIELDS:
        values = [float(record[field]) for record in records if record.get(field) is not None]
        if not values:
            continue
        summary[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }
    for field in ("spell_kinds", "spell_keys", "build_names", "enemy_battle_wins"):
        total = Counter()
        for record in records:
            total.update(record.get(field, {}) or {})
        summary[f"{field}_top"] = dict(total.most_common(12))
    for field in ("built_buildings", "learned_spells", "equipped_hero_items"):
        presence = Counter()
        for record in records:
            presence.update(
                str(item)
                for item in set(record.get(field, []) or [])
                if item is not None and str(item).strip()
            )
        summary[f"{field}_episode_presence"] = {
            key: {"episodes": count, "rate": count / len(records) if records else 0.0}
            for key, count in presence.most_common()
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare winning and losing dragon trajectories.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=105000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.n_envs < 1:
        parser.error("--episodes and --n-envs must be positive")

    raw_vec = SubprocVecEnv([make_env() for _ in range(args.n_envs)], start_method="spawn")
    raw_vec.seed(args.seed)
    env = VecNormalize.load(str(args.vecnormalize.resolve()), raw_vec)
    env.training = False
    env.norm_reward = False
    model = MaskablePPO.load(str(args.model.resolve()))

    trackers = [new_tracker() for _ in range(args.n_envs)]
    records: list[dict[str, Any]] = []
    obs = env.reset()
    try:
        while len(records) < args.episodes:
            masks = get_action_masks(env)
            actions, _ = model.predict(
                obs,
                deterministic=bool(args.deterministic),
                action_masks=masks,
            )
            obs, rewards, dones, infos = env.step(actions)
            for index, info in enumerate(infos):
                update_tracker(trackers[index], info)
                if dones[index] and len(records) < args.episodes:
                    records.append(finish_record(trackers[index], info, float(rewards[index])))
                    trackers[index] = new_tracker()
                    if len(records) % 10 == 0:
                        wins = sum(int(record["is_dragon_win"]) for record in records)
                        print(f"episodes={len(records)} dragon_wins={wins}", flush=True)
    finally:
        env.close()

    wins = [record for record in records if record["is_dragon_win"]]
    non_wins = [record for record in records if not record["is_dragon_win"]]
    result = {
        "model": str(args.model.resolve()),
        "vecnormalize": str(args.vecnormalize.resolve()),
        "evaluation_seed": args.seed,
        "deterministic": bool(args.deterministic),
        "episodes_requested": args.episodes,
        "dragon_wins": len(wins),
        "dragon_win_rate": len(wins) / len(records),
        "summary": {
            "dragon_wins": summarize(wins),
            "non_wins": summarize(non_wins),
        },
        "episodes": records,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
