"""Record actual wins, independently of NumberGrid's shaped reward."""
from stoa import RecordEpisodeMetrics
from stoa.core_wrappers.episode_metrics import RecordEpisodeMetricsState
from stoa.core_wrappers.wrapper import wrapper_state_replace
from stoa.stoa_struct import dataclass
import dataclasses
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_metrics import empty_campaign_counts


@dataclass(custom_replace_fn=wrapper_state_replace)
class CampaignMetricsState(RecordEpisodeMetricsState):
    campaign_counts: jax.Array


def with_campaign_state(state, counts):
    return CampaignMetricsState(**{field.name: getattr(state, field.name)
        for field in dataclasses.fields(RecordEpisodeMetricsState)}, campaign_counts=counts)


class NumberGridEpisodeMetrics(RecordEpisodeMetrics):
    @staticmethod
    def _with_success(timestep):
        metrics = {**timestep.extras['episode_metrics'],
                   'episode_success': timestep.extras['solved_episode']}
        for name in ('battle_transition', 'player_battle_transition', 'enemy_battle_transition', 'automatic_battle_transition', 'battle_victory', 'building_constructed', 'turn_ended', 'rest_penalty'):
            if name in timestep.extras:
                metrics[name] = timestep.extras[name]
        if 'episode_event' in timestep.extras:
            metrics['episode_event'] = timestep.extras['episode_event']
        return timestep.replace(extras={**timestep.extras, 'episode_metrics': metrics})

    def reset(self, rng_key, env_params=None):
        state, timestep = super().reset(rng_key, env_params)
        timestep = self._with_success(timestep)
        if 'campaign_events' in timestep.extras:
            state = with_campaign_state(state, empty_campaign_counts())
            timestep = self._campaign(timestep, state.campaign_counts)
        return state, timestep

    def step(self, state, action, env_params=None):
        following, timestep = super().step(state, action, env_params)
        timestep = self._with_success(timestep)
        if 'campaign_events' in timestep.extras:
            counts = state.campaign_counts + timestep.extras['campaign_events']
            following = with_campaign_state(following, jnp.where(timestep.done(), 0., counts))
            timestep = self._campaign(timestep, counts)
        return following, timestep

    @staticmethod
    def _campaign(timestep, counts):
        metrics = {**timestep.extras['episode_metrics'],
                   'campaign_events': timestep.extras['campaign_events'],
                   'campaign_episode': counts}
        return timestep.replace(extras={**timestep.extras, 'episode_metrics': metrics})
