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


def write_viewer(payload):
    encoded = json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    template = (ROOT/'web/battle.template.html').read_text(encoding='utf-8')
    html = template.replace('__DATA__',encoded)
    html = html.replace('/*__CSS__*/',(ROOT/'web/battle.css').read_text(encoding='utf-8'))
    html = html.replace('/*__APP__*/',(ROOT/'web/battle_app.js').read_text(encoding='utf-8'))
    (ROOT/'viewer.html').write_text(html,encoding='utf-8')


def build(output=None):
    if output is None:
        write_viewer({'map':MAP,'records':[],'result':None})
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

    records=[]
    for index in range(3):
        seed=42+index
        (initial,mask,max_hp),final,(states,actions,rewards,masks,max_hps)=jax.device_get(rollout(jnp.int32(seed),index==0))
        frames=[{'state':state_dict(initial),'action':None,'reward':0.,'total_reward':0.,
                 'action_mask':mask.tolist(),'max_hp':max_hp.tolist()}]
        total=0.
        for t in range(int(final.step_count)):
            action=int(actions[t]);assert frames[-1]['action_mask'][action]
            total+=float(rewards[t])
            state=jax.tree.map(lambda x,t=t:x[t],states)
            frames.append({'state':state_dict(state),'action':action,'reward':float(rewards[t]),
                           'total_reward':total,'action_mask':masks[t].tolist(),'max_hp':max_hps[t].tolist()})
        assert bool(final.done)
        outcome='победа' if bool(final.won) else 'поражение' if bool(final.lost) else 'лимит'
        records.append({'label':f'{"Argmax" if index==0 else "Выборка"} · {outcome} · {int(final.step_count)} шагов',
                        'seed':seed,'policy':'argmax' if index==0 else 'sample','frames':frames})
    payload={'map':game_map,'records':records,'result':result,'source':'Restored PPO checkpoint; JAX rollouts'}
    (output/'trajectories.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    write_viewer(payload)
    print(json.dumps({'viewer':str(ROOT/'viewer.html'),'records':[r['label'] for r in records]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path)
    build(parser.parse_args().results)
