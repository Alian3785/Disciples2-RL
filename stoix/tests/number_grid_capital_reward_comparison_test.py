"""Extended CUDA cases for the paired Capital experiment requested 2026-10-16.

Sources: docs/CAPITAL_REWARD_COMPARISON.md, Python train_campaign.REWARD_CONFIG.
Run only on explicit request through scripts/run_gpu.sh; outside core.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.tests.number_grid_fixtures import compiled_method
from stoix.tests.number_grid_target_level_rewards_test import developed_party, enter, finish, near_enemy


@pytest.fixture(scope='module')
def pair():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    current = load_map('default-capital')
    # This suite isolates the old paired reward profiles, not the later potion shaping.
    current.update(capital_defense_acquisition_reward=0., capital_defense_effect_reward=0.,
                   capital_guardian_round_reward=0., capital_entry_level_reward=0.,
                   capital_guardian_battle_reward=0., capital_incubus_reward=0., capital_incubus_building_reward=0.,
                   capital_guardian_rounds_bonus=0.,
                   recruitment_reward_faction_only=False,
                   episode_timeout_penalty=0., battle_round_limit_penalty=1.)
    python = {**current, 'reward_profile': 'python_reference',
              'reference_preserve_target_reward': True, 'step_cost': 0.}
    envs = [NumberGrid(map_config=config) for config in (current, python)]
    return [(env, compiled_method(env, 'reset')(jax.random.PRNGKey(42))[0]) for env in envs]


def test_target_tactical_reward_is_identical(pair):
    paid = []
    for env, initial in pair:
        state, action = near_enemy(env, developed_party(env, initial), env.objective_enemy)
        battle, _ = enter(env, state, action)
        after, amount = finish(env, battle)
        assert after.won and after.done
        assert float(amount) == pytest.approx(4.+5.*int(battle.army_battle_level_strength))
        paid.append(float(amount))
    assert paid[0] == paid[1]
    assert pair[0][0].observation_size == pair[1][0].observation_size


def test_reference_goal_is_not_paid_twice(pair):
    env, initial = pair[1]
    terminal = initial.replace(won=jnp.bool_(True), done=jnp.bool_(True))
    assert float(rewards.finish(env, initial, terminal)[1]) == 0.
    terminal = terminal.replace(reference_defeated=terminal.reference_defeated.at[env.objective_enemy].set(True))
    after, amount = rewards.finish(env, initial, terminal)
    # The episode-wide defeated-army fraction is a separate reference component.
    assert float(amount) == pytest.approx(5./env.reference_enemy_count)
    assert float(rewards.finish(env, after, after)[1]) == 0.


def test_regular_victories_still_use_the_selected_profile(pair):
    paid = []
    for env, initial in pair:
        state, action = near_enemy(env, developed_party(env, initial), 30)
        battle, _ = enter(env, state, action)
        after, amount = finish(env, battle)
        assert not after.won and not after.done
        assert not jnp.any(after.army_victory_rewarded)
        paid.append(float(amount))
    assert paid[0] == pytest.approx(1.)
    assert paid[1] >= 4.5


@pytest.mark.parametrize('changes', [
    {'reference_preserve_target_reward': 1},
    {'reference_preserve_target_reward': True},
    {'reference_preserve_target_reward': True, 'reward_profile': 'python_reference', 'objective_enemy': None},
])
def test_target_reward_preservation_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError, match='reference_preserve_target_reward'):
        config = {**load_map('default-capital'), **changes}
        if config.get('objective_enemy', 0) is None:
            config.update(objective_reward_hero_only=False, army_victory_level_objective_only=False)
        NumberGrid(map_config=config)
