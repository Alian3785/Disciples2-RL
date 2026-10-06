"""One explicit configuration for the local GTX 1660 Ti, FP32, feedforward PPO."""
from pathlib import Path
from omegaconf import OmegaConf
from stoix.envs.number_grid import MAP

ROOT = Path(__file__).resolve().parent


def make_config(total_timesteps=5_000_000, seed=42, chunk_updates=10, map_config=None):
    game_map = MAP if map_config is None else map_config
    config = OmegaConf.create({
        'arch': {'seed': seed, 'num_envs': 250, 'update_batch_size': 1,
                 'num_updates_per_eval': chunk_updates},
        'env': {'env_name': 'number_grid', 'scenario': {'name': f"NumberGrid-{game_map['size']}x{game_map['size']}"},
                'kwargs': {'map_config': game_map}, 'use_cached_auto_reset': True},
        'logger': {'checkpointing': {'load_model': False}},
        'system': OmegaConf.load(ROOT / 'stoix/configs/system/ppo/ff_ppo.yaml'),
        'network': OmegaConf.load(ROOT / 'stoix/configs/network/mlp.yaml'),
    })
    config.system.rollout_length = 100
    config.system.num_minibatches = 10
    config.network.actor_network.pre_torso.layer_sizes = [128, 128]
    config.network.critic_network.pre_torso.layer_sizes = [128, 128]
    config.network.actor_network.pre_torso.activation = 'tanh'
    config.network.critic_network.pre_torso.activation = 'tanh'
    batch_steps = config.arch.num_envs * config.system.rollout_length
    if total_timesteps <= 0 or total_timesteps % (batch_steps * chunk_updates):
        raise ValueError(f'total_timesteps must be a positive multiple of {batch_steps * chunk_updates}')
    config.arch.num_updates = total_timesteps // batch_steps
    config.arch.total_timesteps = total_timesteps
    config.arch.total_num_envs = config.arch.num_envs
    return config
