"""Export JAX archer battle replays; human play uses serve_number_grid.py."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from stoix.envs.number_grid import MAP, NumberGrid, wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor

ROOT = Path(__file__).resolve().parent


def state_dict(state):
    return {f.name: np.asarray(getattr(state,f.name)).tolist()
            for f in dataclasses.fields(state) if f.name != 'battle_key'}


def pack(payload):
    """Store each distinct frame value once; web/battle_app.js expands shared-replay-values-v1.

    Most state fields repeat between consecutive decisions, so long replays stay loadable.
    """
    pool, index = [], {}

    def ref(value):
        key = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        if key not in index:
            index[key] = len(pool)
            pool.append(value)
        return index[key]

    records = []
    for record in payload['records']:
        frame_keys = sorted({key for frame in record['frames'] for key in frame if key != 'state'})
        state_keys = list(record['frames'][0]['state'])
        records.append(dict(label=record['label'], seed=record['seed'], policy=record['policy'],
            frame_keys=frame_keys, state_keys=state_keys,
            fields=[[ref(frame[key]) if key in frame else -1 for key in frame_keys] for frame in record['frames']],
            states=[[ref(frame['state'][key]) for key in state_keys] for frame in record['frames']]))
    metadata = {key: value for key, value in payload.items() if key != 'records'}
    return dict(format='shared-replay-values-v1', metadata=metadata, pool=pool, records=records)


def write_viewer(payload):
    # Keep the standalone page compact; all replays remain in trajectories.json.
    if payload.get("records"):
        payload = pack({**payload, "records": [min(payload["records"], key=lambda r: len(r["frames"]))]})
    encoded = json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    template = (ROOT/'web/battle.template.html').read_text(encoding='utf-8')
    html = template.replace('__DATA__',encoded)
    html = html.replace('/*__CSS__*/',(ROOT/'web/battle.css').read_text(encoding='utf-8'))
    html = html.replace('/*__APP__*/',(ROOT/'web/battle_app.js').read_text(encoding='utf-8'))
    (ROOT/'viewer.html').write_text(html,encoding='utf-8')


def build(output=None, episodes=3, seeds=None):
    """Argmax with seed 42, then sampled seeds 43...; explicit seeds are all sampled."""
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if output is None:
        env = NumberGrid()
        write_viewer({'map':MAP,'records':[],'result':None, 'construction':env.construction.metadata(), 'turn_rules':env.turn_metadata(), 'combat':env.combat_info, 'capital':env.capital.metadata(), 'potions':env.potion_rules.metadata(), 'equipment':env.item_rules.metadata if env.items_enabled else None, 'sites':env.site_rules.metadata if env.sites_enabled else None, 'recruitment':env.recruitment.metadata if env.recruitment_enabled else None, 'territory':env.territory.metadata if env.territory_enabled else None, 'ruins':env.ruins.metadata if env.ruins_enabled else None, 'spell_research':env.spell_research.metadata if env.spell_research_enabled else None, 'spell_casting':env.casting.metadata if env.spell_casting_enabled else None, 'battle_items':env.battle_item_rules.metadata if env.battle_items_enabled else None, 'unit_swaps':env.unit_swaps.metadata if env.unit_swaps_enabled else None})
        return
    output = Path(output)
    game_map = json.loads((output/'map.json').read_text())
    if not game_map.get('battle_mode'):
        raise ValueError('This exporter requires an archer-battle checkpoint')
    result = json.loads((output/'results.json').read_text())
    checkpoint = (output/'params.msgpack').read_bytes()
    assert hashlib.sha256(checkpoint).hexdigest() == result['checkpoint_sha256']
    params = jax.tree.map(jnp.asarray,serialization.msgpack_restore(checkpoint)['actor_params'])
    config = OmegaConf.load(output/'config.json')
    env = NumberGrid(map_config=game_map)
    policy_env = wrap_wall_action_mask(env)
    actor = FeedForwardActor(torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
                             action_head=hydra.utils.instantiate(config.network.actor_network.action_head,
                                                                action_dim=env.action_space().num_values))

    @jax.jit
    def rollout(seed, greedy):
        state, ts = policy_env.reset(jax.random.PRNGKey(seed))
        initial = (state,ts.observation['action_mask'],env.max_hp(state))
        def step(carry, _):
            state,ts,key = carry
            key,sub = jax.random.split(key)
            dist = actor.apply(params,jax.tree.map(lambda x:x[None],ts.observation))
            action = jnp.where(greedy,dist.mode()[0],dist.sample(seed=sub)[0])
            state,ts = policy_env.step(state,action)
            return (state,ts,key),(state,action,ts.reward,ts.observation['action_mask'],env.max_hp(state))
        final,frames = jax.lax.scan(step,(state,ts,jax.random.PRNGKey(seed+100)),None,length=env.max_steps)
        return initial,final[0],frames

    @jax.jit
    def snapshot_fields(states):
        # Batch display data on the GPU once instead of transferring every
        # frame back to JAX for four small calculations during JSON export.
        return jax.vmap(lambda state: (env.construction.status(state),
            env.rest_penalty(state), env.unit_experience(state),
            env.unit_stats(state), env.capital.quotes(state, env.max_hp(state)),
            env.potion_rules.quotes(state, env.max_hp(state)), env.map_commands(state),
            env.movement_cap(state), env.site_rules.quotes(state) if env.sites_enabled else {},
            env.recruitment.quotes(state) if env.recruitment_enabled else {}, env.territory_quotes(state), env.travel_costs(state),
            env.spell_research.status(state) if env.spell_research_enabled else jnp.zeros(0, jnp.int32),
            env.cast_quotes(state)))(states)

    records=[]
    plan = [(seed, False) for seed in seeds] if seeds else [(42+index, index == 0) for index in range(episodes)]
    for seed, greedy in plan:
        device_rollout = rollout(jnp.int32(seed),greedy)
        (initial,mask,max_hp),final,(states,actions,rewards,masks,max_hps)=jax.device_get(device_rollout)
        statuses,penalties,experiences,stats,quotes,potion_quotes,commands,movement_caps,site_quotes,recruitment_quotes,territory_quotes,travel_costs,spell_statuses,cast_quotes=jax.device_get(snapshot_fields(device_rollout[2][0]))
        frames=[{'state':state_dict(initial),'action':None,'reward':0.,'total_reward':0.,
                 'action_mask':mask.tolist(),'max_hp':max_hp.tolist(),
                  'map_commands':np.asarray(env.map_commands(initial)).tolist(),
                 'building_status':np.asarray(env.construction.status(initial)).tolist(),
                 'spell_status':np.asarray(env.spell_research.status(initial)).tolist() if env.spell_research_enabled else [],
                 'cast_quotes':{k:np.asarray(v).tolist() for k,v in env.cast_quotes(initial).items()} or None,
                 'rest_penalty':float(env.rest_penalty(initial)),
                 'movement_cap':int(env.movement_cap(initial)),
                  'travel_costs':np.asarray(env.travel_costs(initial)).tolist(),
                  'capital_quotes':np.asarray(env.capital.quotes(initial, env.max_hp(initial))).tolist(),
                  'potion_quotes':np.asarray(env.potion_rules.quotes(initial, env.max_hp(initial))).tolist(),
                 'recruitment_quotes':({k:np.asarray(v).tolist() for k,v in env.recruitment.quotes(initial).items()}
                                       if env.recruitment_enabled else None),
                 'site_quotes':({k:np.asarray(v).tolist() for k,v in env.site_rules.quotes(initial).items()}
                                if env.sites_enabled else None),
                 'territory_quotes':{k:np.asarray(v).tolist() for k,v in env.territory_quotes(initial).items()} or None,
                 'unit_experience':np.asarray(env.unit_experience(initial)).tolist(),
                 'unit_stats':np.asarray(env.unit_stats(initial)).tolist() if env.basic_combat else None}]
        total=0.
        for t in range(int(final.step_count)):
            action=int(actions[t]);assert frames[-1]['action_mask'][action]
            total+=float(rewards[t])
            state=jax.tree.map(lambda x,t=t:x[t],states)
            frames.append({'state':state_dict(state),'action':action,'reward':float(rewards[t]),
                           'total_reward':total,'action_mask':masks[t].tolist(),'max_hp':max_hps[t].tolist(),
                           'building_status':statuses[t].tolist(),
                            'spell_status':spell_statuses[t].tolist(),
                            'cast_quotes':{k:v[t].tolist() for k,v in cast_quotes.items()} or None,
                           'rest_penalty':float(penalties[t]),
                           'movement_cap':int(movement_caps[t]),
                            'travel_costs':travel_costs[t].tolist(),
                            'capital_quotes':quotes[t].tolist(),
                            'potion_quotes':potion_quotes[t].tolist(),
                            'map_commands':commands[t].tolist(),
                           'recruitment_quotes':{k:v[t].tolist() for k,v in recruitment_quotes.items()} or None,
                           'site_quotes':{k:v[t].tolist() for k,v in site_quotes.items()} or None,
                           'territory_quotes':{k:v[t].tolist() for k,v in territory_quotes.items()} or None,
                            'unit_experience':experiences[t].tolist(),
                           'unit_stats':stats[t].tolist() if env.basic_combat else None})
        assert bool(final.done)
        outcome='победа' if bool(final.won) else 'поражение' if bool(final.lost) else 'лимит'
        defeated=int(np.sum(~np.asarray(final.alive) & np.asarray(final.scenario_spawned)))
        records.append({'label':f'{"Argmax" if greedy else "Выборка"} · {outcome} · {int(final.step_count)} шагов'
                                 f' · {defeated} армий · награда {total:.1f}',
                        'seed':seed,'policy':'argmax' if greedy else 'sample','frames':frames})
    payload={'map':game_map,'records':records,'result':result,'source':'Restored PPO checkpoint; JAX rollouts',
             'construction':env.construction.metadata(), 'turn_rules':env.turn_metadata(), 'combat':env.combat_info, 'capital':env.capital.metadata(), 'potions':env.potion_rules.metadata(), 'equipment':env.item_rules.metadata if env.items_enabled else None, 'sites':env.site_rules.metadata if env.sites_enabled else None, 'recruitment':env.recruitment.metadata if env.recruitment_enabled else None, 'territory':env.territory.metadata if env.territory_enabled else None, 'ruins':env.ruins.metadata if env.ruins_enabled else None, 'spell_research':env.spell_research.metadata if env.spell_research_enabled else None, 'spell_casting':env.casting.metadata if env.spell_casting_enabled else None, 'battle_items':env.battle_item_rules.metadata if env.battle_items_enabled else None, 'unit_swaps':env.unit_swaps.metadata if env.unit_swaps_enabled else None}
    (output/'trajectories.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    write_viewer(payload)
    print(json.dumps({'viewer':str(ROOT/'viewer.html'),'records':[r['label'] for r in records]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--episodes', type=int, default=3,
                        help='Replay count: first is argmax, subsequent episodes sample actions')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Record sampled episodes with these seeds instead of the default replays')
    args = parser.parse_args()
    build(args.results, episodes=args.episodes, seeds=args.seeds)
