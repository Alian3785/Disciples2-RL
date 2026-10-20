"""Prove trade_train winnable using legal actions in the unchanged CUDA environment.

This deterministic verification planner is separate from training. It never
changes state fields, supplies demonstrations, trains weights or measures speed.
The JSON contains the seed and every action, so the witness can be replayed.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, NumberGrid, action_names, load_map
from stoix.envs.number_grid_limits import MAX_RESOURCE


class Walkthrough:
    def __init__(self, seed):
        if jax.default_backend() != 'gpu':
            raise RuntimeError('CUDA is required; use scripts/run_gpu.sh')
        self.game_map = load_map('trade_train')
        self.env = NumberGrid(map_config=self.game_map)
        self.seed = seed
        self.names = action_names(self.game_map)
        self.step = jax.jit(self.env.step)
        self.view = jax.jit(lambda s: (self.env.action_mask(s), self.env.max_hp(s),
                                      jnp.all(jnp.isfinite(self.env.observation(s)))))
        self.state, _ = jax.jit(self.env.reset)(jax.random.PRNGKey(seed))
        self.actions, self.events, self.level_reward_payments = [], [], []
        self.reward = 0.
        self.sync()
        self.blocked = jax.device_get(self.env.territory.blocked_grid).reshape(self.env.size, self.env.size)
        self.directions = self.env.directions.tolist()
        self.training_quotes = jax.jit(self.env.site_rules.training_quotes)

        @jax.jit
        def earn(state, target):
            def condition(carry):
                s, count, _, valid = carry
                return (s.gold < target) & ~s.done & (count < self.game_map['max_steps']) & valid

            def advance(carry):
                s, count, reward, valid = carry
                legal = self.env.action_mask(s)[REST]
                after, ts = self.env.step(s, jnp.int32(REST))
                finite = jnp.isfinite(ts.reward) & jnp.all(jnp.isfinite(self.env.observation(after)))
                return after, count+1, reward+ts.reward, valid & legal & finite

            return jax.lax.while_loop(condition, advance, (state, jnp.int32(0), jnp.float32(0), jnp.bool_(True)))

        self.earn_on_device = earn

    def sync(self):
        self.host = jax.device_get(self.state)
        self.mask, self.max_hp, finite = jax.device_get(self.view(self.state))
        assert finite, 'Non-finite observation'

    def act(self, action):
        if isinstance(action, str):
            action = self.names.index(action)
        assert self.mask[action], f'Illegal action {self.names[action]} at {self.host.position}'
        assert not self.host.done, 'Action after terminal state'
        before = self.host
        self.state, timestep = self.step(self.state, jnp.int32(action))
        reward = float(timestep.reward)
        assert bool(jnp.isfinite(timestep.reward)), 'Non-finite reward'
        self.reward += reward
        self.actions.append(int(action))
        self.sync()
        if self.env.army_victory_level_reward:
            paid = self.host.army_victory_rewarded & ~before.army_victory_rewarded
            defeated = before.alive & ~self.host.alive
            assert not bool((paid & ~defeated).any()), 'Army level bonus without victory'
            for enemy in range(self.env.num_opponents):
                if paid[enemy]:
                    strength = int(before.army_battle_level_strength)
                    if not before.in_battle:
                        strength = sum(max(int(level)-1, 0) for uid, level, hp in zip(
                            before.unit_ids[:6], before.unit_levels[:6], before.hp[:6]) if uid and hp > 0)
                    self.level_reward_payments.append(dict(enemy=enemy+1, decision=len(self.actions),
                        strength=strength, bonus=self.env.army_victory_level_reward*strength))
            if not self.host.in_battle or self.host.done:
                assert int(self.host.army_battle_level_strength) == 0, 'Stale battle strength'
        assert not self.host.lost, f'Lost after {len(self.actions)} actions'
        assert len(self.actions) < self.game_map['max_steps'], 'Planner exceeded episode limit'

    def event(self, kind, **extra):
        party = [dict(slot=i, unit=self.env.progression.rows[int(uid)]['name'],
                      level=int(self.host.unit_levels[i]), xp=int(self.host.unit_xp[i]),
                      hp=int(self.host.hp[i]), max_hp=int(self.max_hp[i]))
                 for i, uid in enumerate(self.host.unit_ids[:6]) if uid]
        row = dict(kind=kind, decisions=len(self.actions), day=int(self.host.day),
                   gold=int(self.host.gold), party=party, **extra)
        self.events.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def move_to(self, goals):
        goals = {tuple(p) for p in goals}
        start = tuple(self.host.position)
        queue = deque([start])
        previous = {start: None}
        enemies = {tuple(p) for p, alive in zip(self.host.enemy_positions, self.host.alive) if alive}
        found = None
        while queue:
            here = queue.popleft()
            if here in goals:
                found = here
                break
            for action, (dr, dc) in enumerate(self.directions):
                there = (here[0]+dr, here[1]+dc)
                if (there not in previous and there not in enemies
                        and 0 < there[0] < self.env.size-1 and 0 < there[1] < self.env.size-1
                        and not self.blocked[there]):
                    previous[there] = (here, action)
                    queue.append(there)
        assert found is not None, f'Unreachable: {goals}'
        route = []
        while previous[found] is not None:
            found, action = previous[found]
            route.append(action)
        for action in reversed(route):
            if not self.mask[action]:
                assert self.host.movement_points == 0, 'Path disagrees with movement mask'
                self.act(REST)
            self.act(action)
            assert not self.host.in_battle, 'Unexpected encounter on route'

    def site(self, kind):
        self.move_to(self.env.site_rules.approaches[kind].tolist())

    def recover(self):
        self.move_to([self.game_map['agent_position']])
        for slot, uid in enumerate(self.host.unit_ids[:6]):
            if uid and self.host.hp[slot] <= 0:
                self.act(f'revive_{slot}')
        for _ in range(20):
            if (self.host.hp[:6] >= self.max_hp[:6]).all():
                return
            self.act(REST)
        raise AssertionError('Rest did not heal living army')

    def train(self, slots):
        self.site('trainer')
        for slot in slots:
            if not self.host.unit_ids[slot]:
                continue
            action = self.names.index(f'train_{slot}')
            # Pay in batches within the resource cap, using normal capital
            # income and legal trainer actions; never inject state or resources.
            while True:
                quote = jax.device_get(self.training_quotes(self.state))[slot]
                cost = int(quote[0]*quote[2])
                if cost <= 0:
                    break
                self.earn(min(cost, MAX_RESOURCE))
                if not self.mask[action]:
                    break
                previous_xp = int(self.host.unit_xp[slot])
                self.act(action)
                assert int(self.host.unit_xp[slot]) > previous_xp, 'Training made no progress'
        self.event('trained')

    def earn(self, target):
        if self.host.gold >= target:
            return
        # Batch repetitive verification actions on CUDA; still call the normal
        # step and check its mask/reward/observation for every individual REST.
        self.state, count, reward, valid = self.earn_on_device(self.state, jnp.int32(target))
        assert bool(valid), 'Illegal REST or non-finite transition'
        self.actions.extend([REST]*int(count))
        self.reward += float(reward)
        self.sync()
        assert not self.host.done and self.host.gold >= target, 'Could not earn requested gold'

    def buy_final_preparation(self):
        self.site('mercenary')
        self.earn(1800)
        self.act('hire_mercenary_watcher')
        self.site('merchant')
        self.earn(3400)
        for key in ('highfather', 'invulnerability', 'invulnerability', 'celerity', 'celerity'):
            self.act('buy_'+key)
        self.event('final_preparation_bought')

    def fight(self, enemy):
        r, c = self.host.enemy_positions[enemy]
        self.move_to([(r-dr, c-dc) for dr, dc in self.directions])
        offset = [int(r-self.host.position[0]), int(c-self.host.position[1])]
        action = self.directions.index(offset)
        if not self.mask[action]:
            self.act(REST)
        if enemy == 2:
            # Apply temporary buffs after the final travel/rest. Wight is immune
            # to weapons and death: keep air/fire attackers alive and acting early.
            self.act('potion_highfather_3')
            for slot in (3, 5):
                self.act(f'potion_invulnerability_{slot}')
                self.act(f'potion_celerity_{slot}')
        self.event('battle_started', enemy=enemy+1)
        self.act(action)
        for _ in range(600):
            if not self.host.in_battle or self.host.done:
                break
            targets = [i for i in range(6) if self.mask[SHOOT+i]]
            if targets:
                action = SHOOT+min(targets, key=lambda i: self.host.hp[6+i])
            else:
                action = DEFEND if self.mask[DEFEND] else CONTINUE
            self.act(action)
        assert not self.host.alive[enemy], f'Enemy {enemy+1} survived'
        self.event('battle_won', enemy=enemy+1)

    def run(self):
        self.event('reset')
        temple = next(i for i, b in enumerate(self.env.construction.rows) if b['name'] == 'Храм')
        self.act(f'build_{temple}')
        self.site('mercenary')
        for unit in ('defender_of_faith', 'white_mage', 'g000uu0176'):
            self.act('hire_mercenary_'+unit)
        self.event('hired')
        for index, enemy in enumerate((5, 4, 0, 3, 1, 2)):
            self.train([1, 0, 3, 4, 2, 5])
            self.recover()
            if index >= 2 and not self.host.unit_ids[2]:
                self.site('mercenary')
                action = self.names.index('hire_mercenary_centaur_savage')
                self.earn(1800)
                if self.mask[action]:
                    self.act(action)
                    self.event('reinforced')
            if enemy == 2:
                self.buy_final_preparation()
            self.fight(enemy)
        assert self.host.won and self.host.done and not self.host.alive.any()
        if self.env.army_victory_level_reward:
            assert self.host.army_victory_rewarded.tolist() == [True]*6
            assert len(self.level_reward_payments) == 6
        return dict(map='trade_train', seed=self.seed, backend=str(jax.devices()[0]),
                    map_sha256=hashlib.sha256(Path('maps/trade_train.json').read_bytes()).hexdigest(),
                    won=True, decisions=len(self.actions), reward=self.reward,
                    action_names=self.names, actions=self.actions, events=self.events,
                    training_transitions=0, state_injections=0,
                    army_victory_rewarded=self.host.army_victory_rewarded.tolist(),
                    level_reward_payments=self.level_reward_payments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    planner = Walkthrough(args.seed)
    try:
        result = planner.run()
    except Exception:
        planner.event('failed', enemy_hp=planner.host.hp[6:].tolist(),
                      round=int(planner.host.round), in_battle=bool(planner.host.in_battle))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.failed.json').write_text(json.dumps(dict(
            seed=args.seed, actions=planner.actions, events=planner.events), ensure_ascii=False, indent=2)+'\n')
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(won=True, decisions=result['decisions'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
