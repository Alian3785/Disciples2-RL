"""Invoke the unchanged PPO trainer with phase timing, no external tracking."""
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    profile_path = Path(sys.argv[1])
    train_args = sys.argv[2:]
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['MPLBACKEND'] = 'Agg'
    from sb3_contrib import MaskablePPO
    original_learn = MaskablePPO.learn
    original_collect = MaskablePPO.collect_rollouts
    original_train = MaskablePPO.train
    profile = {'rollout_seconds': 0.0, 'optimizer_seconds': 0.0,
               'variant': os.environ['CAMPAIGN_OBSERVATION_VERSION']}

    def collect(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_collect(self, *args, **kwargs)
        profile['rollout_seconds'] += time.perf_counter() - start
        return result

    def train(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_train(self, *args, **kwargs)
        profile['optimizer_seconds'] += time.perf_counter() - start
        return result

    def learn(self, *args, **kwargs):
        profile.update(observation_size=self.observation_space.shape[0],
                       policy_parameters=sum(p.numel() for p in self.policy.parameters()))
        start = time.perf_counter()
        result = original_learn(self, *args, **kwargs)
        profile.update(learn_seconds=time.perf_counter() - start,
                       actual_steps=int(self.num_timesteps), status='completed')
        profile['fps'] = self.num_timesteps / profile['learn_seconds']
        profile_path.write_text(json.dumps(profile, indent=2) + '\n')
        return result

    MaskablePPO.collect_rollouts = collect
    MaskablePPO.train = train
    MaskablePPO.learn = learn
    sys.argv = [str(Path(__file__).with_name('train_campaign.py')), *train_args]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
