"""Record the existing visualize_campaign Matplotlib figures to a portable MP4.

Requires imageio-ffmpeg. No display server or alternate renderer is used.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

os.environ['MPLBACKEND'] = 'Agg'
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import imageio_ffmpeg
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

import visualize_campaign as visualization
from battle_env import BattleEnv
from maps.super_last_stand import SCHEDULED_WAVES


class VideoRecorder:
    """Save canvas pixels verbatim, padding the smaller battle window in white."""

    def __init__(self, output, fps, playback_speed=1.0):
        self.output = output
        self.fps = fps
        self.playback_speed = playback_speed
        self.width, self.height = 1550, 1000
        self.pending = None
        self.hold_seconds = 0.0
        self.frames = 0
        self.draws = 0
        self.previews = set()
        self.process = None

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(self.output.stem + '.partial.mp4')
        self.encoder_log = self.output.with_suffix('.encoder.log').open('w')
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgba',
            '-s', f'{self.width}x{self.height}', '-r', str(self.fps * self.playback_speed), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(self.partial),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.encoder_log)
        return self

    def flush(self):
        if self.pending is None:
            return
        count = max(1, round(self.hold_seconds * self.fps))
        for _ in range(count):
            self.process.stdin.write(self.pending)
        self.frames += count
        self.pending = None
        self.hold_seconds = 0.0

    def capture(self, canvas):
        self.flush()
        rgba = np.asarray(canvas.buffer_rgba())
        height, width = rgba.shape[:2]
        if width > self.width or height > self.height:
            raise ValueError(f'Canvas {width}x{height} exceeds video size')
        frame = np.full((self.height, self.width, 4), 255, dtype=np.uint8)
        left, top = (self.width - width) // 2, (self.height - height) // 2
        frame[top:top + height, left:left + width] = rgba
        self.pending = frame.tobytes()
        self.draws += 1
        kind = 'map' if width == self.width else 'battle'
        if kind not in self.previews:
            from PIL import Image
            Image.fromarray(frame).save(self.output.with_name(self.output.stem + f'_{kind}.png'))
            self.previews.add(kind)
        if self.draws % 100 == 0:
            print(f'[Recording] {self.draws} drawings, {self.frames / (self.fps * self.playback_speed):.1f}s video', flush=True)

    def sleep(self, seconds):
        # Preserve the visualizer's intended pauses in the video, without sleeping.
        self.hold_seconds += max(0.0, float(seconds))

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.sleep(3.0)
                self.flush()
        finally:
            self.process.stdin.close()
            code = self.process.wait()
            self.encoder_log.close()
        if exc_type is None:
            if code:
                raise RuntimeError(f'FFmpeg exited with code {code}; see encoder log')
            self.partial.replace(self.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--vecnormalize', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map', default='super_last_stand')
    parser.add_argument('--objective', choices=('cities', 'dragon', 'blue_dragon', 'full_party', 'waves'))
    parser.add_argument('--no-boss-roster', action='store_true', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--expected-episode', type=Path)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--playback-speed', type=float, default=1.0,
                        help='Speed of the entire video, including pauses (0.5 = twice as slow)')
    parser.add_argument('--delay', type=float, default=0.3)
    parser.add_argument('--battle-speed', type=float, default=1.0)
    parser.add_argument('--max-steps', type=int, default=5000)
    args = parser.parse_args()
    expected = json.loads(args.expected_episode.read_text()) if args.expected_episode else None
    for filename in (args.model, args.vecnormalize):
        if not Path(filename).is_file():
            parser.error(f'File does not exist: {filename}')
    if args.output.suffix.lower() != '.mp4' or args.fps < 1 or args.max_steps < 1:
        parser.error('Output must be .mp4; fps and max-steps must be positive')
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error('playback-speed must be finite and positive')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    battle_seeds = random.Random(args.seed)
    original_battle_init = BattleEnv.__init__
    original_reset = visualization.CampaignEnv.reset
    original_step = visualization.CampaignEnv.step
    original_draw = FigureCanvasAgg.draw
    wave_by_enemy = {int(enemy): int(wave['wave_index'])
                     for wave in SCHEDULED_WAVES for enemy in wave['enemy_ids']} if args.map == 'super_last_stand' else {}
    summary = dict(model=str(Path(args.model).resolve()), seed=args.seed,
                   deterministic=args.deterministic, map=args.map, objective=args.objective,
                   use_boss_starting_roster=False, observation_version='local5', steps=0,
                   reward=0.0, waves=0, wave_stacks=0, reached_wave=0,
                   renderer='visualize_campaign.py (unmodified canvas pixels)')

    def seeded_battle_init(battle, *a, **kw):
        original_battle_init(battle, *a, **kw)
        battle.seed(battle_seeds.getrandbits(64))

    def seeded_reset(env, *a, **kw):
        if kw.get('seed') is not None:
            random.seed(kw['seed'])
            battle_seeds.seed(kw['seed'])
        return original_reset(env, *a, **kw)

    def tracked_step(env, action):
        obs, reward, terminated, truncated, info = original_step(env, action)
        summary['steps'] += 1
        summary['reward'] += float(reward)
        summary['waves'] = max(summary['waves'], int(info.get('waves_defeated_count', 0)))
        summary['wave_stacks'] = max(summary['wave_stacks'], int(info.get('wave_stacks_defeated_count', 0)))
        for key in ('enemy_id', 'target_enemy_id', 'terminal_defeat_enemy_id'):
            summary['reached_wave'] = max(summary['reached_wave'], wave_by_enemy.get(int(info.get(key) or 0), 0))
        if not (terminated or truncated) and summary['steps'] >= args.max_steps:
            truncated = True
            info = dict(info, campaign_result='eval_timeout')
        if terminated or truncated:
            summary['result'] = info.get('campaign_result', 'timeout' if truncated else 'other')
            summary['terminal_enemy'] = info.get('terminal_defeat_enemy_id')
        return obs, reward, terminated, truncated, info

    context = nullcontext(SimpleNamespace(frames=0, width=1550, height=1000, sleep=lambda seconds: None)) if args.verify_only else VideoRecorder(args.output, args.fps, args.playback_speed)
    with context as recorder:
        def recorded_draw(canvas):
            if not args.verify_only:
                original_draw(canvas)
                recorder.capture(canvas)

        with patch.object(BattleEnv, '__init__', seeded_battle_init), \
             patch.object(visualization.CampaignEnv, 'reset', seeded_reset), \
             patch.object(visualization.CampaignEnv, 'step', tracked_step), \
             patch.object(FigureCanvasAgg, 'draw', recorded_draw), \
             patch.object(visualization, 'time', SimpleNamespace(sleep=recorder.sleep)), \
             patch.object(visualization, 'display', None), \
             patch('builtins.input', return_value=''):
            visualization.run_campaign_visualization(
                model_path=args.model, vecnormalize_path=args.vecnormalize,
                map_name=args.map, seed=args.seed, deterministic=args.deterministic,
                delay=args.delay, battle_speed=args.battle_speed, max_grid_steps=3000,
                campaign_objective=args.objective, use_boss_starting_roster=False)
        if expected:
            assert summary['result'] == expected['result'], (summary, expected)
            assert summary['steps'] == expected['length'], (summary, expected)
            assert summary['waves'] == expected['waves'], (summary, expected)
            assert abs(summary['reward'] - expected['reward']) < 0.01, (summary, expected)
            summary['evaluation_match'] = True
    summary.update(frames=recorder.frames, fps=args.fps * args.playback_speed,
                   playback_speed=args.playback_speed,
                   duration_seconds=recorder.frames / (args.fps * args.playback_speed),
                   resolution=[recorder.width, recorder.height],
                   output=str(args.output.resolve()))
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('VIDEO SAVED', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
