import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from maps import available_maps, get_map


def _make_small_env(**kwargs) -> CampaignEnv:
    params = dict(map_name="small", Realcapital=2, log_enabled=False)
    params.update(kwargs)
    return CampaignEnv(**params)


def test_small_map_is_registered_and_halved():
    assert "small" in available_maps()
    small = get_map("small")
    default = get_map("default")
    assert small.play_grid_size == 24
    # Примерно вдвое меньше объектов каждого типа.
    assert len(small.enemy_stacks) == 37
    assert len(default.enemy_stacks) == 76  # 75 уникальных id, стек 22 задан дважды
    assert len(small.chests) == len(default.chests) // 2
    assert len(small.mana_sources) == len(default.mana_sources) // 2
    assert len(small.obstacle_blocks) == (len(default.obstacle_blocks) + 1) // 2
    assert len(small.ruin_rewards) == 2
    assert len(small.merchant_sites) == 1
    assert len(small.mercenary_sites) == 1
    # Все четыре вида маны сохранены.
    assert {kind for kind, _ in small.mana_sources} == {"infernal", "life", "death", "runes"}


def test_small_map_env_plays_on_24_grid_with_scaled_castle():
    env = _make_small_env()
    assert env.grid_size == 24
    assert env.campaign_objective == "cities"
    assert tuple(env.grid_env.start_position) == (2, 13)
    # Столичная хилка совпадает со стартом героя.
    assert env.castle_heal_tiles[0] == (2, 13)
    assert len(env.castle_heal_tiles) == 3  # столица + два целевых города
    assert all(0 <= x < 24 and 0 <= y < 24 for x, y in env._static_enemy_positions.values())


def test_small_map_untouchables_are_present():
    env = _make_small_env()
    pos = env._static_enemy_positions
    # Дракон-цель, столица Империи, оба целевых города с гарнизонами.
    for enemy_id in (31, 75, 34, 68, 35, 69):
        assert enemy_id in pos, f"enemy {enemy_id} missing"
    # Гарнизоны делят клетку с защитниками городов, города — на своих хилках.
    assert pos[34] == pos[68] == (14, 16)
    assert pos[35] == pos[69] == (20, 12)
    assert set(env.FINAL_OBJECTIVE_CITIES) == {"Порт Полонис", "Соругирилла"}
    assert env.scripted_capital_bot_config_enabled is True
    assert env.empire_territory_enabled is True
    assert env.empire_territory_source_tile == pos[75]
    # Обе «охотничьи» цели валидны на этой карте (дракон присутствует).
    _make_small_env(campaign_objective="dragon")
    _make_small_env(campaign_objective="blue_dragon")


def test_small_map_has_no_unintended_position_collisions():
    env = _make_small_env()
    pos = env._static_enemy_positions
    allowed_pairs = [{34, 68}, {35, 69}]
    by_tile = Counter(pos.values())
    for tile, count in by_tile.items():
        if count > 1:
            ids = {eid for eid, p in pos.items() if p == tile}
            assert ids in allowed_pairs, f"unintended collision at {tile}: {sorted(ids)}"


def test_small_map_uses_template_starting_roster():
    env = _make_small_env()
    assert env.use_boss_starting_roster is False
    env.reset(seed=0)
    names = [
        str(u.get("name", "")).strip()
        for u in env.blue_team_state
        if str(u.get("name", "")).strip().lower() != "пусто"
    ]
    assert names == ["Одержимый", "Герцог", "Одержимый", "Сектант"]
    # Явный аргумент по-прежнему сильнее дефолта карты.
    boss_env = _make_small_env(use_boss_starting_roster=True)
    assert boss_env.use_boss_starting_roster is True


def test_default_map_roster_and_grid_defaults_unchanged():
    env = CampaignEnv(Realcapital=2, log_enabled=False)
    assert env.grid_size == 48
    assert env.use_boss_starting_roster is True


def test_small_map_random_masked_rollout_is_stable():
    env = _make_small_env(freeze_dynamic_action_layout=True)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for step_idx in range(150):
        mask = env.compute_action_mask()
        valid_actions = np.flatnonzero(mask)
        assert valid_actions.size > 0, f"empty action mask at step {step_idx}"
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(valid_actions)))
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            env.reset(seed=step_idx + 1)
