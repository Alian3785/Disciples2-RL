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

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid import MAP, NumberGrid, CONTINUE

ROOT = Path(__file__).resolve().parent


class GameService:
    def __init__(self):
        self.env = NumberGrid()
        self.reset = jax.jit(self.env.reset)
        self.advance = jax.jit(self.env.step)
        self.sessions = OrderedDict()
        self.lock = threading.Lock()
        state, _ = self.reset(jax.random.PRNGKey(0))
        jax.block_until_ready(self.advance(state, jnp.int32(2)))

    def snapshot(self, state, total_reward):
        state = jax.device_get(state)
        values = {f.name: np.asarray(getattr(state, f.name)).tolist()
                  for f in dataclasses.fields(state) if f.name != 'battle_key'}
        return {'state': values, 'total_reward': total_reward,
                'battle_max_rounds': self.env.max_rounds,
                'max_steps': self.env.max_steps,
                'action_mask': np.asarray(self.env.action_mask(state)).tolist(),
                'max_hp': np.asarray(self.env.max_hp(state)).tolist()}

    def create(self, seed):
        with self.lock:
            state, _ = self.reset(jax.random.PRNGKey(seed))
            token = secrets.token_urlsafe(24)
            self.sessions[token] = (state, 0.)
            while len(self.sessions) > 64:
                self.sessions.popitem(last=False)
            return {'session': token, 'map': self.env.map_config,
                    'snapshot': self.snapshot(state, 0.), 'events': []}

    def act(self, token, action):
        with self.lock:
            if token not in self.sessions:
                raise KeyError('Сессия истекла. Начните новую игру.')
            state, total = self.sessions[token]
            if bool(state.done):
                raise ValueError('Игра завершена. Начните новую игру.')
            if not (0 <= action < 18 and bool(self.env.action_mask(state)[action])):
                raise ValueError('Сейчас это действие недоступно.')
            events = []
            state, ts = self.advance(state, jnp.int32(action))
            total += float(ts.reward)
            events.append(self.snapshot(state, total))
            # Only the human UI advances scripted turns automatically. The PPO
            # environment counts and observes every unit transition separately.
            for _ in range(32):
                if bool(state.done) or not bool(state.in_battle) or not bool(self.env.action_mask(state)[CONTINUE]):
                    break
                state, ts = self.advance(state, jnp.int32(CONTINUE))
                total += float(ts.reward)
                events.append(self.snapshot(state, total))
            self.sessions[token] = (state, total)
            self.sessions.move_to_end(token)
            return {'session': token, 'snapshot': self.snapshot(state, total), 'events': events}


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
                return self.send_json(200, {'ok':True, 'environment':MAP['name'], 'backend':jax.default_backend()})
            if path == '/api/info':
                return self.send_json(200, {'map':MAP})
            filename = 'viewer.html' if path in ('/', '/viewer.html') else path.lstrip('/')
            if filename not in ('viewer.html', 'web/battle_app.js', 'web/battle.css'):
                return self.send_error(404)
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
                    result = service.create(seed)
                elif self.path == '/api/step':
                    action = body.get('action')
                    if type(action) is not int:
                        raise ValueError('Неверное действие.')
                    result = service.act(body.get('session'), action)
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
    args = parser.parse_args()
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Start with scripts/run_gpu.sh to use the CUDA environment.')
    service = GameService()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, args.port))
    print(f'NumberGrid human mode: http://127.0.0.1:{args.port}/viewer.html', flush=True)
    server.serve_forever()
