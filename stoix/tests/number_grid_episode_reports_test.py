"""Extended CUDA checks for episode diagnostics; run only on explicit request.

Sources: the existing progression, unit-swap, map-summon and autoreset rules.
"""
import copy
import gzip
import json

import chex
import jax
import jax.numpy as jnp
import pytest
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import AddRNGKey

from numbergrid_episode_reports import EpisodeReports, read_episode_batch
from stoix.envs.number_grid import REST
from stoix.envs.number_grid_episode_reports import (
    HEADER, WIDTH, compact_episode_events, episode_event, permanent_party,
)
from stoix.wrappers.number_grid_metrics import NumberGridEpisodeMetrics
from stoix.wrappers.number_grid_reset import NumberGridBattleAutoReset


@pytest.fixture(autouse=True, scope='module')
def cuda_only():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh; no CPU fallback'


def decode(event):
    row = jax.device_get(event['data']).tolist()
    p = len(HEADER)
    return dict(zip(HEADER, row[:p])), row[p:p+24], row[p+24:p+54], row[p+54:p+60]


def test_awarded_xp_survives_promotion_reset_and_rejected_repeat(current_game):
    env, initial, _, _ = current_game
    before = initial.replace(in_battle=jnp.bool_(True), enemy=jnp.int32(0), step_count=jnp.int32(7))
    after = before.replace(in_battle=jnp.bool_(False), step_count=jnp.int32(8),
        alive=before.alive.at[0].set(False), unit_levels=before.unit_levels.at[0].add(1),
        unit_xp=before.unit_xp.at[0].set(0), last_xp=jnp.zeros(12, jnp.int32).at[0].set(157))
    measure = jax.jit(lambda a,b,c,valid,counts: episode_event(env,a,b,c,jnp.int32(REST),valid,counts))
    event = measure(before,before,after,jnp.bool_(True),jnp.array([1,0,0,1],jnp.int32))
    header, _, final, gains = decode(event)
    assert header['method'] == header['battle_result'] == 1
    assert header['defeated_enemy'] == 0 and gains[0] == 157
    assert final[12] == 0  # Current XP reset; awarded XP is retained independently.
    rejected = after.replace(step_count=jnp.int32(9))
    assert not measure(after,after,rejected,jnp.bool_(False),jnp.zeros(4,jnp.int32))['emit']


def test_trainer_xp_and_whole_column_swap_preserve_identity(current_game):
    env, state, _, _ = current_game
    fighter = int(state.unit_ids[0])
    large = next(i for i, profile in enumerate(env.combat_info['catalogue'])
                 if profile and profile['size'] == 2)
    before = state.replace(step_count=jnp.int32(4),
        unit_ids=state.unit_ids.at[:6].set(jnp.array([large,fighter,0,0,fighter,0])),
        unit_levels=state.unit_levels.at[:6].set(jnp.array([1,1,0,0,1,0])),
        unit_xp=state.unit_xp.at[:6].set(jnp.array([10,20,0,0,30,0])),
        hp=state.hp.at[:6].set(jnp.array([1,1,0,0,1,0])))
    action = jnp.int32(env.unit_swaps.start + env.unit_swaps.pairs.index((0,1)))

    @jax.jit
    def swap(s):
        moved, _ = env.unit_swaps.apply(s, action, jnp.bool_(False))
        return episode_event(env,s,moved,moved.replace(step_count=s.step_count+1),
                             action,jnp.bool_(True),jnp.zeros(4,jnp.int32))

    event = swap(before)
    assert decode(event)[3] == [0]*6  # A changed slot must not manufacture XP.
    p = len(HEADER) + 60
    assert jax.device_get(event['data'][p:p+6]).tolist() == [1,0,2,4,3,5]
    trained = before.replace(step_count=jnp.int32(5),
        unit_xp=before.unit_xp.at[1].add(45), last_xp=jnp.full(12,999,jnp.int32))
    measure = jax.jit(lambda a,b,valid: episode_event(env,a,b,b,jnp.int32(REST),
                                                    valid,jnp.zeros(4,jnp.int32)))
    assert decode(measure(before,trained,jnp.bool_(True)))[3] == [0,45,0,0,0,0]
    assert not measure(trained,trained.replace(step_count=jnp.int32(6)),jnp.bool_(False))['emit']


def test_temporary_rosters_do_not_replace_permanent_fighters(current_game):
    _, state, _, _ = current_game
    original = jax.jit(permanent_party)(state)
    copied = state.replace(copied=state.copied.at[0].set(True),
        native_ids=state.native_ids.at[0].set(state.unit_ids[0]),
        native_levels=state.native_levels.at[0].set(state.unit_levels[0]),
        native_xp=state.native_xp.at[0].set(state.unit_xp[0]),
        unit_ids=state.unit_ids.at[0].set(state.unit_ids[1]),
        unit_levels=state.unit_levels.at[0].set(90), unit_xp=state.unit_xp.at[0].set(999))
    chex.assert_trees_all_equal(jax.jit(permanent_party)(copied), original)
    summon = state.replace(summon_battle=jnp.bool_(True),
        spell_party=jnp.stack((state.unit_ids[:6],state.unit_levels[:6],state.unit_xp[:6],state.hp[:6])),
        unit_ids=jnp.zeros_like(state.unit_ids), unit_xp=jnp.full_like(state.unit_xp,999))
    chex.assert_trees_all_equal(jax.jit(permanent_party)(summon), original)


def test_newly_spawned_immediate_victory_and_summon_defeat(current_game):
    env, state, _, _ = current_game
    before = state.replace(alive=state.alive.at[0].set(False))
    world = before.replace(alive=before.alive.at[0].set(True), enemy=jnp.int32(0), in_battle=jnp.bool_(True))
    after = world.replace(alive=world.alive.at[0].set(False), in_battle=jnp.bool_(False), step_count=jnp.int32(1))
    measure = jax.jit(lambda a,b,c,counts: episode_event(env,a,b,c,jnp.int32(REST),jnp.bool_(True),counts))
    assert decode(measure(before,world,after,jnp.array([1,0,0,0])))[0]['defeated_enemy'] == 0
    world = world.replace(summon_battle=jnp.bool_(True))
    survived = state.replace(step_count=jnp.int32(1), last_xp=jnp.ones(12,jnp.int32)*999)
    header, _, _, gains = decode(measure(state,world,survived,jnp.array([0,1,0,0])))
    assert header['outcome'] == 0 and header['method'] == header['battle_result'] == 2
    assert gains == [0]*6


def test_compaction_keeps_all_events_and_replica_stream_order():
    shape = (2,3,2,3,4)  # device, update, batch, time, env; 144 events, no 100-event cap
    indices = jnp.arange(144).reshape(shape)
    data = jnp.zeros(shape+(WIDTH,),jnp.int32).at[...,0].set(indices)
    count, packets = jax.jit(compact_episode_events)({
        'episode_event':dict(emit=jnp.ones(shape,bool),data=data),
        'episode_return':indices.astype(jnp.float32)})
    batch = read_episode_batch(packets,int(count))
    expected = [((((d*3+u)*2+b)*3+t)*4+n)
                for u in range(3) for t in range(3) for d in range(2) for b in range(2) for n in range(4)]
    assert int(count) == 144
    assert batch['data'][:,0].tolist() == expected
    assert batch['stream'].tolist() == list(range(16))*9
    assert batch['offset'].tolist() == list(range(144))


def test_empty_batch_has_no_padding_events():
    count, packets = jax.jit(compact_episode_events)({
        'episode_event': dict(emit=jnp.zeros((3,2),bool),data=jnp.zeros((3,2,WIDTH),jnp.int32)),
        'episode_return': jnp.zeros((3,2))})
    assert int(count) == 0
    assert all(not values for values in read_episode_batch(packets,int(count)).values())


def packet(step, old, new, *, outcome=0, gains=None, order=None, enemy=-1, result=0):
    header = [step,1,outcome,enemy,1 if result else 0,result,enemy if result==1 else -1,2,3,0]
    data = header + sum(old,[]) + sum(new,[]) + [100]*6 + (gains or [0]*6) + (order or list(range(6))) + [1]*6
    return dict(data=[data],returns=[9.5],stream=[0],offset=[step-1])


def test_writer_identity_xp_history_hire_and_loss_across_batches(tmp_path):
    combat = dict(catalogue=[None,dict(name='Боец',key='fighter'),dict(name='Ветеран',key='veteran')],
                  enemies=[[dict(name='Дракон')]+[None]*5])
    writer = EpisodeReports(tmp_path,dict(name='fixture',reference_enemy_ids=[31]),combat)
    old = [[1,1,0,0,0,0],[1,1,0,0,0,0],[0]*6,[1,1,0,0,0,0]]
    promoted = [[2,1,0,0,0,0],[2,1,0,0,0,0],[0,17,0,0,0,0],[1,1,0,0,0,0]]
    writer.consume(packet(1,old,promoted,gains=[150,17,0,0,0,0],enemy=0,result=1))
    order = [1,0,2,3,4,5]
    moved = [[row[i] for i in order] for row in promoted]
    writer.consume(packet(2,promoted,moved,order=order))
    dismissed = [row.copy() for row in moved]
    for row in dismissed:
        row[1] = 0
    writer.consume(packet(3,moved,dismissed))
    hired = [row.copy() for row in dismissed]
    hired[0][1] = hired[1][1] = hired[3][1] = 1
    writer.consume(packet(4,dismissed,hired))
    dead = [row.copy() for row in hired]
    dead[3] = [0]*6
    writer.consume(packet(5,hired,dead,outcome=2,enemy=0,result=2))
    assert writer.path.stat().st_size == 0  # No writes between interval summaries.
    summary = writer.write_summary(expected_completed=1,final=True)
    with gzip.open(writer.path, 'rt', encoding='utf-8') as stream:
        episode = json.load(stream)
    assert summary['episodes_file_bytes'] == writer.path.stat().st_size
    assert summary['episodes_file_bytes'] < summary['episodes_uncompressed_bytes']
    assert [u['fighter_id'] for u in episode['final_party']] == [2,3]
    assert [u['xp_received'] for u in episode['final_party']] == [17,0]
    assert episode['fighters'][0]['xp_received'] == 150
    assert not episode['fighters'][0]['in_final_party']
    assert len(episode['upgrades']) == 1 and episode['upgrades'][0]['fighter_id'] == 1
    assert episode['defeated_armies'][0]['army']['reference_id'] == 31
    assert episode['defeat']['opponent']['units'][0]['name'] == 'Дракон'


def test_compressed_report_retains_every_interval_and_final_partial_interval(tmp_path):
    combat = dict(catalogue=[None,dict(name='Боец',key='fighter')],
                  enemies=[[dict(name='Дракон')]+[None]*5])
    writer = EpisodeReports(tmp_path,dict(name='fixture'),combat)
    party = [[1,0,0,0,0,0],[1,0,0,0,0,0],[0]*6,[1,0,0,0,0,0]]
    for expected in (1,2,3):
        previous_size = writer.path.stat().st_size
        writer.consume(packet(1,party,party,outcome=1,enemy=0,result=1))
        assert writer.path.stat().st_size == previous_size
        summary = writer.write_summary(expected_completed=expected,final=expected==3)
        with gzip.open(writer.path, 'rt', encoding='utf-8') as stream:
            episodes = [json.loads(line) for line in stream]
        assert [e['episode'] for e in episodes] == list(range(1,expected+1))
        assert summary['completed_episodes'] == expected
    size = writer.path.stat().st_size
    writer.write_summary(expected_completed=3,final=True)
    assert writer.path.stat().st_size == size  # Empty final flush cannot duplicate records.


def test_autoreset_retains_terminal_report_and_preserves_transition(current_game):
    raw, initial, plain_step, _ = current_game
    reporting = copy.copy(raw)
    reporting.episode_reports = True
    reporting.max_steps = 1
    regular = copy.copy(raw)
    regular.max_steps = 1
    state, baseline = jax.jit(regular.step)(initial,jnp.int32(REST))
    observed, ts = jax.jit(reporting.step)(initial,jnp.int32(REST))
    chex.assert_trees_all_equal(state,observed)
    chex.assert_trees_all_equal(baseline.observation,ts.observation)
    assert baseline.reward == ts.reward and baseline.discount == ts.discount
    env = VmapWrapper(NumberGridBattleAutoReset(NumberGridEpisodeMetrics(AddRNGKey(reporting)),next_obs_in_extras=True))
    wrapped,_ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(7),2))
    advance = jax.jit(env.step)
    for _ in range(2):
        wrapped,ts = advance(wrapped,jnp.full(2,REST,jnp.int32))
        events = ts.extras['episode_metrics']['episode_event']
        assert jnp.all(events['emit'])
        assert jnp.all(events['data'][:,0] == 1) and jnp.all(events['data'][:,2] == 3)
        assert jnp.all(wrapped.step_count == 0)
