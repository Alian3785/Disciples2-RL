"""Record actual wins, independently of NumberGrid's shaped reward."""
from stoa import RecordEpisodeMetrics


class NumberGridEpisodeMetrics(RecordEpisodeMetrics):
    @staticmethod
    def _with_success(timestep):
        metrics = {**timestep.extras['episode_metrics'],
                   'episode_success': timestep.extras['solved_episode']}
        for name in ('battle_transition', 'player_battle_transition', 'enemy_battle_transition', 'battle_victory'):
            if name in timestep.extras:
                metrics[name] = timestep.extras[name]
        return timestep.replace(extras={**timestep.extras, 'episode_metrics': metrics})

    def reset(self, rng_key, env_params=None):
        state, timestep = super().reset(rng_key, env_params)
        return state, self._with_success(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = super().step(state, action, env_params)
        return state, self._with_success(timestep)
