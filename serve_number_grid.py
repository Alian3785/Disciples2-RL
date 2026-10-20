"""Local human-mode server using the very same JAX environment as PPO.

The browser only sends actions and renders snapshots; it never simulates combat.
Bind to loopback and serialize actions per session. No training runs in this server.
"""
import argparse
from collections import OrderedDict
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

import hashlib
import os
import sys

# The map fixes ACTION_NAMES at import time, so select it before importing the env.
_selected = argparse.ArgumentParser(add_help=False)
_selected.add_argument('--map')
_selected = _selected.parse_known_args(sys.argv[1:])[0].map
if _selected:
    os.environ['NUMBER_GRID_MAP'] = _selected

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid import ACTION_NAMES, MAP, NumberGrid
from stoix.envs.number_grid_buildings import DEFAULT_FACTION, FACTIONS
from stoix.envs.number_grid_lords import LORDS

ROOT = Path(__file__).resolve().parent


DIRECTIONS = dict(N='↑ север', NE='↗ северо-восток', E='→ восток', SE='↘ юго-восток',
                  S='↓ юг', SW='↙ юго-запад', W='← запад', NW='↖ северо-запад')
SIMPLE = dict(defend='защита', wait='ждать', retreat='отступать', rest='отдых · закончить ход',
              transform_fenrir='Дух Фенрира')


def action_label(action):
    name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f'action_{action}'
    if name in DIRECTIONS:
        return DIRECTIONS[name]
    if name in SIMPLE:
        return SIMPLE[name]
    if name == 'continue':
        return 'продолжить'
    kind, _, rest = name.partition('_')
    index = rest.rsplit('_', 1)[-1]
    slot = int(index) + 1 if index.isdigit() else index
    if kind == 'shoot':
        return f'действие по цели {slot}'
    if kind == 'build':
        return f'постройка (слот {slot})'
    if kind == 'heal':
        return f'храм: лечить бойца {slot}'
    if kind == 'revive':
        return f'храм: воскресить бойца {slot}'
    if kind == 'copy':
        return f'облик союзника {slot}'
    if kind == 'potion':
        return f'зелье {rest.rsplit("_", 1)[0]} → боец {slot}'
    if kind == 'equip':
        return f'боевой слот героя: {rest}'
    if kind in ('item1', 'item2'):
        side = 'союзник' if rest.startswith('ally') else 'враг'
        return f'боевой предмет {kind[-1]} → {side} {slot}'
    return name


class Agent:
    """Trained PPO actor used only for hints and agent moves in human mode."""

    def __init__(self, results):
        import hydra
        from flax import serialization
        from omegaconf import OmegaConf
        from stoix.networks.base import FeedForwardActor
        results = Path(results)
        result = json.loads((results / 'results.json').read_text(encoding='utf-8'))
        checkpoint = (results / 'params.msgpack').read_bytes()
        if hashlib.sha256(checkpoint).hexdigest() != result['checkpoint_sha256']:
            raise ValueError('params.msgpack does not match results.json')
        # The whole map defines observations and actions, not just its name.
        if json.loads((results / 'map.json').read_text(encoding='utf-8')) != MAP:
            raise ValueError('The checkpoint was trained on a different map.')
        config = OmegaConf.load(results / 'config.json')
        self.params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(checkpoint)['actor_params'])
        self.actor = FeedForwardActor(
            torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
            action_head=hydra.utils.instantiate(config.network.actor_network.action_head,
                                                action_dim=len(ACTION_NAMES)))
        evaluation = result.get('final_evaluation') or {}
        self.info = dict(training_steps=result['training_steps'],
                         steps_per_second=result['mean_steps_per_second'],
                         argmax_wins=evaluation.get('successes'),
                         argmax_episodes=evaluation.get('episodes'))
        self.policies = {}

    def policy(self, env):
        if id(env) not in self.policies:
            @jax.jit
            def choose(state):
                mask = env.action_mask(state)
                obs = {'observation': env.observation(state)[None], 'action_mask': mask[None]}
                probs = jax.nn.softmax(self.actor.apply(self.params, obs).logits[0])
                probs = jnp.where(mask, probs, 0.)
                return jnp.argmax(probs), probs
            self.policies[id(env)] = choose
        return self.policies[id(env)]


class GameService:
    def __init__(self, agent=None):
        self.environments = {}
        self.sessions = OrderedDict()
        self.lock = threading.Lock()
        self.agent = agent
        self.env, reset, advance = self.environment(MAP.get('faction', DEFAULT_FACTION))
        state, _ = reset(jax.random.PRNGKey(0))
        jax.block_until_ready(advance(state, jnp.int32(2)))
        if agent is not None:
            jax.block_until_ready(agent.policy(self.env)(state))

    def environment(self, faction, lord_type=MAP.get('lord_type', 'warrior')):
        if faction not in FACTIONS:
            raise ValueError('Неизвестная фракция.')
        if not isinstance(lord_type, str) or lord_type not in LORDS:
            raise ValueError('Неизвестный тип правителя.')
        key = (faction, lord_type)
        if key not in self.environments:
            env = NumberGrid(map_config={**MAP, 'faction': faction, 'lord_type': lord_type})
            self.environments[key] = (env, jax.jit(env.reset), jax.jit(env.step))
        return self.environments[key]

    def snapshot(self, env, state, total_reward):
        state = jax.device_get(state)
        values = {f.name: np.asarray(getattr(state, f.name)).tolist()
                  for f in dataclasses.fields(state) if f.name != 'battle_key'}
        return {'state': values, 'total_reward': total_reward,
                'battle_max_rounds': env.max_rounds,
                'forced_retreat': bool(env.round_limit_retreat(state)) and not bool(state.done),
                'battle_round_limit_penalty': env.round_limit_penalty,
                'max_steps': env.max_steps,
                'action_mask': np.asarray(env.action_mask(state)).tolist(),
                'map_commands': np.asarray(env.map_commands(state)).tolist(),
                'max_hp': np.asarray(env.max_hp(state)).tolist(),
                'building_status': np.asarray(env.construction.status(state)).tolist(),
                'spell_status': np.asarray(env.spell_research.status(state)).tolist() if env.spell_research_enabled else [],
                'cast_quotes': {k: np.asarray(v).tolist() for k, v in env.cast_quotes(state).items()} or None,
                'rest_penalty': float(env.rest_penalty(state)),
                'movement_cap': int(env.movement_cap(state)),
                'travel_costs': np.asarray(env.travel_costs(state)).tolist(),
                'capital_quotes': np.asarray(env.capital.quotes(state, env.max_hp(state))).tolist(),
                'potion_quotes': np.asarray(env.potion_rules.quotes(state, env.max_hp(state))).tolist(),
                'recruitment_quotes': ({k: np.asarray(v).tolist() for k,v in env.recruitment.quotes(state).items()}
                                      if env.recruitment_enabled else None),
                'site_quotes': ({k: np.asarray(v).tolist() for k,v in env.site_rules.quotes(state).items()}
                                if env.sites_enabled else None),
                'territory_quotes': ({k: np.asarray(v).tolist() for k,v in env.territory_quotes(state).items()}
                                     if env.territory_enabled else None),
                'battle_item_quotes': ({k: np.asarray(v).tolist() for k,v in env.battle_item_rules.quotes(env, state).items()}
                                       if env.battle_items_enabled else None),
                'unit_experience': np.asarray(env.unit_experience(state)).tolist(),
                'unit_stats': (np.asarray(env.unit_stats(state)).tolist()
                               if env.basic_combat else None)}

    def create(self, seed, faction=DEFAULT_FACTION, lord_type=MAP.get('lord_type', 'warrior')):
        with self.lock:
            env, reset, _ = self.environment(faction, lord_type)
            state, _ = reset(jax.random.PRNGKey(seed))
            token = secrets.token_urlsafe(24)
            self.sessions[token] = ((faction, lord_type), state, 0.)
            while len(self.sessions) > 64:
                self.sessions.popitem(last=False)
            return {'session': token, 'map': env.map_config, 'construction': env.construction.metadata(),
                    'turn_rules': env.turn_metadata(), 'combat': env.combat_info,
                    'capital': env.capital.metadata(), 'potions': env.potion_rules.metadata(),
                    'equipment': env.item_rules.metadata if env.items_enabled else None,
                    'recruitment': env.recruitment.metadata if env.recruitment_enabled else None,
                    'territory': env.territory.metadata if env.territory_enabled else None,
                    'ruins': env.ruins.metadata if env.ruins_enabled else None,
                    'spell_research': env.spell_research.metadata if env.spell_research_enabled else None,
                    'spell_casting': env.casting.metadata if env.spell_casting_enabled else None,
                    'battle_items': env.battle_item_rules.metadata if env.battle_items_enabled else None,
                    'unit_swaps': env.unit_swaps.metadata if env.unit_swaps_enabled else None,
                    'sites': env.site_rules.metadata if env.sites_enabled else None,
                    'snapshot': self.snapshot(env, state, 0.), 'events': []}

    def act(self, token, action):
        with self.lock:
            if token not in self.sessions:
                raise KeyError('Сессия истекла. Начните новую игру.')
            key, state, total = self.sessions[token]
            env, _, advance = self.environment(*key)
            if bool(state.done):
                raise ValueError('Игра завершена. Начните новую игру.')
            if not (0 <= action < env.action_space().num_values and bool(env.action_mask(state)[action])):
                raise ValueError('Сейчас это действие недоступно.')
            events = []
            state, ts = advance(state, jnp.int32(action))
            total += float(ts.reward)
            events.append({**self.snapshot(env, state, total), 'reward': float(ts.reward)})
            # The same JAX step used by PPO already advanced all scripted turns.
            self.sessions[token] = (key, state, total)
            self.sessions.move_to_end(token)
            return {'session': token, 'snapshot': self.snapshot(env, state, total), 'events': events}

    def suggest(self, token):
        if self.agent is None:
            raise ValueError('Сервер запущен без модели (--model).')
        with self.lock:
            if token not in self.sessions:
                raise KeyError('Сессия истекла. Начните новую игру.')
            key, state, _ = self.sessions[token]
            env, _, _ = self.environment(*key)
            if bool(state.done):
                raise ValueError('Игра завершена. Начните новую игру.')
            action, probs = jax.device_get(self.agent.policy(env)(state))
            action = int(action)
            return {'action': action, 'label': action_label(action),
                    'probability': float(probs[action])}


def make_handler(service, port):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/health':
                return self.send_json(200, {'ok':True, 'environment':MAP['name'], 'map':os.environ.get('NUMBER_GRID_MAP', 'current'), 'backend':jax.default_backend(),
                                            'agent': service.agent.info if service.agent else None})
            if path == '/api/info':
                return self.send_json(200, {'map':MAP})
            filename = 'viewer.html' if path in ('/', '/viewer.html') else path.lstrip('/')
            if filename not in ('viewer.html', 'web/battle_app.js', 'web/battle.css'):
                return self.send_error(404)
            if filename == 'viewer.html' and service.agent is None:
                # A newly selected manual map must not inherit a saved model's
                # replay or its embedded, older UI. Metadata arrives via reset.
                payload = json.dumps({'map': MAP, 'records': [], 'result': None},
                                     ensure_ascii=False).replace('<', '\\u003c')
                html = (ROOT/'web/battle.template.html').read_text(encoding='utf-8')
                html = html.replace('__DATA__', payload)
                html = html.replace('<style>/*__CSS__*/</style>', '<link rel="stylesheet" href="/web/battle.css">')
                html = html.replace('<script>/*__APP__*/</script>', '<script src="/web/battle_app.js"></script>')
                content = html.encode('utf-8')
            else:
                content = (ROOT / filename).read_bytes()
            kind = 'text/html' if filename.endswith('.html') else 'text/css' if filename.endswith('.css') else 'text/javascript'
            self.send_response(200)
            self.send_header('Content-Type', kind+'; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.send_json(403, {'error':'Недопустимый источник запроса.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Неверный размер запроса.')
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/reset':
                    seed = body.get('seed', 42)
                    if type(seed) is not int or not 0 <= seed < 2**32:
                        raise ValueError('Seed должен быть целым числом от 0 до 2³²−1.')
                    faction = body.get('faction', MAP.get('faction', DEFAULT_FACTION))
                    if not isinstance(faction, str):
                        raise ValueError('Фракция должна быть строкой.')
                    result = service.create(seed, faction, body.get('lord_type', MAP.get('lord_type', 'warrior')))
                elif self.path == '/api/step':
                    action = body.get('action')
                    if type(action) is not int:
                        raise ValueError('Неверное действие.')
                    result = service.act(body.get('session'), action)
                elif self.path == '/api/agent':
                    result = service.suggest(body.get('session'))
                else:
                    return self.send_error(404)
                self.send_json(200, result)
            except KeyError as exc:
                self.send_json(410, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                self.send_json(400, {'error': str(exc)})

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--map', help='current, default, default-green-dragon, default-blue-dragon, default-capital, small, small-green-dragon, small-blue-dragon, super_last_stand, siege_train, wotans_retribution, trade_train, mirror_match (mirrow_match), bethrezens_retreat, or a map JSON path')
    parser.add_argument('--model', help='Results folder with params.msgpack: enables agent hints and moves')
    parser.add_argument('--allow-cpu', action='store_true', help='Run human mode on the JAX CPU backend')
    args = parser.parse_args()
    if jax.default_backend() != 'gpu' and not args.allow_cpu:
        raise RuntimeError('Start with scripts/run_gpu.sh to use the CUDA environment.')
    service = GameService(Agent(args.model) if args.model else None)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, args.port))
    print(f'NumberGrid human mode: http://127.0.0.1:{args.port}/viewer.html', flush=True)
    server.serve_forever()
