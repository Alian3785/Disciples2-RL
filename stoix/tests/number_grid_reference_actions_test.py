"""Reference combat actions and enemy targeting, compiled on CUDA only."""
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state
import copy

import pytest
import chex
import jax
import jax.numpy as jnp

from stoix.envs.number_grid import MAP, NumberGrid, SHOOT, DEFEND


def scalar_attack_cases(env,states,actions,rolls):
    """Mechanics cases share their scalar graph; Chex tests vmap separately."""
    attack = compiled_method(env,'_battle_step')
    results = [attack(jax.tree.map(lambda x,i=i:x[i],states),actions[i],states.battle_key[i],rolls[i])
               for i in range(len(actions))]
    return jax.tree.map(lambda *xs:jnp.stack(xs),*results)


def scalar_battle(env):
    # Batched equivalence has dedicated Chex and summoning tests. These tables
    # vary mechanics, so share the scalar executable with the follow-up turns.
    return lambda states,actions,keys,rolls: scalar_attack_cases(env,states,actions,rolls)


def test_archer_lowest_hp_nonimmune_wards_and_random_ties():
    config = copy.deepcopy(MAP)
    config['hero_units'] = 5
    config['hero_roster'] = ['possessed', 'duke', 'possessed', 'cultist', 'cultist', None]
    config['hero_combat_stats'] = [dict(armor=90), {}, dict(immunities=['weapon']),
                                   dict(protections=['weapon']), {}]
    env = NumberGrid(map_config={**config,'initial_potions':{},'chests':[]})
    start, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    start = compiled_method(env,'_begin_battle')(start.replace(enemy=jnp.int32(2))).replace(actor=jnp.int32(6))
    # Lowest HP wins over an unarmoured guaranteed kill. Immunity is skipped;
    # an unused ward is a valid target. Death and escape both remove a target.
    health = jnp.array([[10, 12, 1, 20, 30, 0], [0, 0, 1, 20, 30, 0],
                        [0, 0, 1, 0, 0, 0], [10, 10, 1, 20, 30, 0],
                        [10, 10, 1, 20, 30, 0], [10, 12, 1, 20, 30, 0]], jnp.int32)
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (6,)+a.shape), start)
    states = states.replace(hp=states.hp.at[:, :6].set(health),
                            escaped=states.escaped.at[5, 0].set(True))
    keys = jnp.zeros((6, 6)).at[3, :2].set(jnp.array([.1, .9])).at[4, :2].set(jnp.array([.9, .1]))
    actions = compiled_method(env,'_enemy_action',batched=True)(states, keys)
    chex.assert_trees_all_equal(actions, jnp.array([SHOOT, SHOOT+3, DEFEND, SHOOT, SHOOT+1, SHOOT+1]))


def test_archer_reference_minimum_hp_replacement():
    from stoix.envs.number_grid_combat import UNITS
    profile = UNITS[MAP['enemy_rosters'][2][0]]
    assert profile['name'] == 'Гоблин лучник'
    assert tuple(profile[key] for key in ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')) == (40, 15, 80, 0, 50)
    assert profile['attack_type'] == 'weapon' and profile['size'] == 1


def test_mage_hero_falloff_counts_living_immune_targets_but_not_dead_or_escaped(elemental_attack_game):
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env, initial, _ = elemental_attack_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','apprentice','cultist',None])
    initial = initial.replace(unit_ids=initial.unit_ids.at[3].set(env.progression.enemy_ids[0,2]))
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = enemy_roster_state(env,start,['archer']*6)
    start = start.replace(unit_ids=start.unit_ids.at[7].set(env.progression.enemy_ids[0,3])
        .at[10].set(env.progression.enemy_ids[0,4]))
    start = start.replace(hp=start.hp.at[6:].set(jnp.array([0,100,45,45,45,45])),
                          escaped=start.escaped.at[8].set(True))
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (2,)+a.shape), start)
    states = states.replace(actor=jnp.array([3,4], jnp.int32))
    # Identical rolls: the hero uses 80/70/60/50%; the ordinary mage uses 80%.
    rolls = jnp.zeros(env.random_size).at[12:24].set(.65)
    result, _ = scalar_battle(env)(
        states, jnp.full(2,SHOOT+1,jnp.int32), states.battle_key,
        jnp.broadcast_to(rolls,(2,env.random_size)))
    chex.assert_trees_all_equal(result.hp[:, 6:], jnp.array([[0,100,45,30,45,45], [0,85,45,30,30,30]]))
    chex.assert_trees_all_equal(result.wards_used, jnp.zeros((2,12), jnp.uint32))
    chex.assert_trees_all_equal(result.last_immune, jnp.array([1 << 7, 0], jnp.uint32))
    chex.assert_trees_all_equal(result.last_target, jnp.array([-1,-1]))


@pytest.mark.core
def test_mage_catalogue_native_stats_and_single_enemy_cast(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import UNITS
    unit = UNITS['apprentice']
    assert MAP['enemy_rosters'][3][3] == 'apprentice'
    assert tuple(unit[k] for k in ('max_hp','damage','accuracy','initiative','exp_kill','exp_required')) == (35,15,80,40,15,75)
    assert unit['attack_type'] == 'air'
    env, initial, _, attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(3))).replace(actor=jnp.int32(9))
    result, _ = attack(state, jnp.int32(CONTINUE), state.battle_key, jnp.zeros(env.random_size))
    chex.assert_trees_all_equal(result.hp[:6], jnp.array([105,135,105,30,0,0]))
    assert result.last_damage == 60 and result.turn_phase[9] == 2


def test_wolf_lord_fenrir_form_damage_mask_and_reversion_before_xp(current_game):
    from stoix.envs.number_grid import FENRIR, CONTINUE, TRANSFORMED
    from stoix.envs.number_grid_combat import HP, DAMAGE, INITIATIVE, MELEE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env, initial, step, attack = current_game
    initial = hero_roster_state(env,initial,['wolf_lord','duke','possessed','cultist','cultist',None])
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    # A native Water-immune large unit replaces the custom profile. Its paired
    # rear slot stays empty; damage and form checks reuse the shared CUDA graph.
    start = enemy_roster_state(env,start,['ismir_son','squire','squire',None,'archer','archer'])
    start = start.replace(hp=start.hp.at[0].set(112).at[6].set(200))
    start = start.replace(enemy_initial_hp=start.hp[6:])
    mask = compiled_method(env,'action_mask')(start)
    assert mask[FENRIR] and mask[SHOOT+5] and len(mask) == 244
    transformed, _ = attack(start, jnp.int32(FENRIR),start.battle_key,jnp.zeros(env.random_size))
    assert transformed.fenrir[0] and transformed.hp[0] == 137
    assert transformed.last_event == TRANSFORMED and transformed.turn_phase[0] == 2
    chex.assert_trees_all_equal(transformed.hp[1:], start.hp[1:])
    values = compiled_method(env,'unit_stats')(transformed)
    assert values[0, HP] == 275 and values[0, DAMAGE] == 90 and values[0, INITIATIVE] == 65
    assert env.unit_traits(transformed)[0, 0] == MELEE and env.unit_traits(transformed)[0, 1] == 1
    actor = transformed.replace(actor=jnp.int32(0))
    assert not compiled_method(env,'action_mask')(actor)[FENRIR] and not compiled_method(env,'action_mask')(actor)[SHOOT+5]
    invalid, _ = step(actor, jnp.int32(FENRIR))
    chex.assert_trees_all_equal(invalid.hp, actor.hp)
    chex.assert_trees_all_equal(invalid.battle_key, actor.battle_key)
    hit, _ = attack(actor, jnp.int32(SHOOT), actor.battle_key, jnp.zeros(env.random_size))
    assert hit.hp[6] == 110 and hit.last_damage == 90  # weapon bypasses water immunity
    # An outer Witch form must restore the existing Fenrir form, not the base
    # Wolf Lord. Battle completion then unwinds both layers before awarding XP.
    nested_start = transformed.replace(actor=jnp.int32(11),
        hp=transformed.hp.at[1:6].set(jnp.minimum(transformed.hp[1:6],100)),
        unit_ids=transformed.unit_ids.at[11].set(env.progression.ids['witch']))
    nested, _ = attack(nested_start, jnp.int32(CONTINUE), nested_start.battle_key,
                       jnp.zeros(env.random_size).at[36:].set(.99))
    assert nested.imp[0] and nested.fenrir[0] and compiled_method(env,'unit_stats')(nested)[0, HP] == 275
    restored = compiled_method(env,'_start_activation')(nested.replace(actor=jnp.int32(0),
        activation_done=nested.activation_done.at[0].set(False)), jnp.float32(.995))
    assert not restored.imp[0] and restored.fenrir[0] and compiled_method(env,'unit_stats')(restored)[0, DAMAGE] == 90
    # No enemy-script transformation: the reference chooses the water AoE.
    enemy = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(4))).replace(actor=jnp.int32(10))
    assert env._enemy_action(enemy, jnp.zeros(6)) < DEFEND
    cast, _ = attack(enemy, jnp.int32(CONTINUE), enemy.battle_key, jnp.zeros(env.random_size))
    chex.assert_trees_all_equal(cast.hp[:6], initial.hp[:6]-jnp.array([40,40,40,40,40,0]))
    assert not jnp.any(cast.fenrir)
    # Restore the natural form before awarding enough XP for a new level.
    winning = nested.replace(actor=jnp.int32(3),
        hp=nested.hp.at[6:].set(jnp.array([1,0,0,0,0,0])),
        unit_xp=nested.unit_xp.at[0].set(2024))
    won, _ = attack(winning, jnp.int32(SHOOT), winning.battle_key, jnp.zeros(env.random_size))
    assert not won.in_battle and not jnp.any(won.fenrir) and not jnp.any(won.imp)
    assert won.unit_levels[0] == 5 and won.hp[0] == 250 and won.unit_xp[0] == 0
    assert compiled_method(env,'unit_stats')(won)[0, DAMAGE] == 40
    # Escaped units and corpses also lose the temporary form.
    fleeing = transformed.replace(actor=jnp.int32(0),
        retreating=transformed.retreating.at[0].set(True),
        escaped=transformed.escaped.at[1:6].set(True))
    fled, _ = attack(fleeing, jnp.int32(CONTINUE), fleeing.battle_key, jnp.zeros(env.random_size))
    assert not fled.in_battle and not jnp.any(fled.fenrir) and fled.hp[0] == 112
    corpse = winning.replace(hp=winning.hp.at[0].set(0))
    dead, _ = attack(corpse, jnp.int32(SHOOT), corpse.battle_key, jnp.zeros(env.random_size))
    assert not jnp.any(dead.fenrir) and dead.hp[0] == 0 and dead.unit_levels[0] == 4
    limited = nested.replace(actor=jnp.int32(0), retreating=nested.retreating.at[0].set(True),
                             escaped=nested.escaped.at[1:6].set(True))
    stopped, _ = attack(limited, jnp.int32(CONTINUE), limited.battle_key, jnp.zeros(env.random_size))
    assert not stopped.in_battle and not jnp.any(stopped.imp) and not jnp.any(stopped.fenrir)
    assert stopped.hp[0] == 112 and stopped.unit_levels[0] == 4


def test_teurg_shatter_uses_secondary_source_ignores_power_and_stacks_after_damage(elemental_attack_game):
    from stoix.envs.number_grid_combat import ARMOR
    env,initial,_ = elemental_attack_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(37)))
    start = hero_roster_state(env,start,['possessed','duke','possessed','teurg','cultist',None])
    # Shared fixture includes a Teurg variant with Water/zero-accuracy
    # secondary attack and a distinct target marked as a capital guard.
    start = start.replace(actor=jnp.int32(3),
        unit_ids=start.unit_ids.at[3].set(env.progression.enemy_ids[1,0]))
    attack, stats = compiled_method(env,'_battle_step'), compiled_method(env,'unit_stats')
    first, _ = attack(start, jnp.int32(SHOOT), start.battle_key, jnp.zeros(env.random_size))
    chex.assert_trees_all_equal(first.hp[6:], jnp.array([72,100,72,100,72,72]))
    chex.assert_trees_all_equal(stats(first)[6:, ARMOR], jnp.array([30,30,30,30,15,30]))
    assert first.last_immune == (1 << 6)+(1 << 7)
    assert first.last_ward == (1 << 8)+(1 << 9)
    second, _ = attack(first.replace(actor=jnp.int32(3)), jnp.int32(SHOOT), first.battle_key, jnp.zeros(env.random_size))
    chex.assert_trees_all_equal(second.hp[6:], jnp.array([44,100,44,72,38,44]))
    chex.assert_trees_all_equal(stats(second)[6:, ARMOR], jnp.array([30,30,15,15,0,30]))
    dying = start.replace(hp=start.hp.at[10].set(1))
    killed, _ = attack(dying, jnp.int32(SHOOT), dying.battle_key, jnp.zeros(env.random_size))
    assert killed.hp[10] == 0 and killed.armor_shreds[10] == 0
    miss, _ = attack(start, jnp.int32(SHOOT), start.battle_key, jnp.full(env.random_size,.999))
    chex.assert_trees_all_equal(miss.hp, start.hp)
    assert not jnp.any(miss.armor_shreds) and not jnp.any(miss.wards_used)
    reset = compiled_method(env,'_begin_battle')(second)
    assert not jnp.any(reset.armor_shreds)
    chex.assert_trees_all_equal(stats(reset)[6:, ARMOR], jnp.full(6, 30.))


def test_witch_small_and_large_forms_zero_damage_recovery_and_wait(current_game):
    from stoix.envs.number_grid import IMP_TRANSFORMED, WAIT
    from stoix.envs.number_grid_combat import HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE
    env,initial,_,_ = current_game
    start = hero_roster_state(env,initial,['possessed','duke','possessed','sorceress','cultist',None])
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    start = enemy_roster_state(env,compiled_method(env,'_begin_battle')(start.replace(enemy=jnp.int32(21))),
        ['titan','squire','titan',None,'acolyte',None]).replace(actor=jnp.int32(3))
    attack = compiled_method(env,'_battle_step')
    rolls = jnp.zeros(env.random_size).at[36:].set(.99)
    first, _ = attack(start, jnp.int32(SHOOT), start.battle_key, rolls)
    chex.assert_trees_all_equal(first.hp, start.hp)
    assert first.imp[6] and first.last_event == IMP_TRANSFORMED
    values = compiled_method(env,'unit_stats')(first)
    chex.assert_trees_all_equal(values[6], jnp.array([250,30,70,0,50], jnp.float32))
    assert env.unit_sizes(first)[6] == 2 and first.hp[9] == 0
    second, _ = attack(first.replace(actor=jnp.int32(3)), jnp.int32(SHOOT+1), first.battle_key, rolls)
    values = compiled_method(env,'unit_stats')(second)
    assert second.imp[7] and second.last_damage == 0
    assert values[7, HP] == 100 and values[7, DAMAGE] == 20
    assert values[7, ACCURACY] == 80 and values[7, ARMOR] == 0 and values[7, INITIATIVE] == 30
    # Recovery precedes the action; age uses battle rounds, including after Wait.
    fresh = second.replace(actor=jnp.int32(7), activation_done=second.activation_done.at[7].set(False))
    recover = compiled_method(env,'_start_activation')
    restored = recover(fresh, jnp.float32(.895))
    stayed = recover(fresh, jnp.float32(.885))
    assert not restored.imp[7] and stayed.imp[7]
    waited = stayed.replace(turn_phase=stayed.turn_phase.at[7].set(1),priority=stayed.priority.at[7].set(-1.))
    assert recover(waited, jnp.float32(0)).imp[7]
    next_round = waited.replace(activation_done=waited.activation_done.at[7].set(False))
    assert not recover(next_round, jnp.float32(.995)).imp[7]
    assert not compiled_method(env,'action_mask')(second.replace(actor=jnp.int32(3)))[WAIT]  # already acted
    healer, _ = attack(start, jnp.int32(SHOOT+4), start.battle_key, rolls)
    assert healer.imp[10] and env._enemy_action(healer.replace(actor=jnp.int32(10)), jnp.zeros(6)) == DEFEND
    # Retained permanent identity and footprint are visible with the status flags.
    chex.assert_trees_all_equal(second.unit_ids, start.unit_ids)
    obs = compiled_method(env,'observation')(second)
    assert obs.shape == (1843,) and int(round(float(obs[460+5*env.num_opponents])*255)) & 1
    ending = second.replace(actor=jnp.int32(4), hp=second.hp.at[6:].set(jnp.array([1,1,1,0,1,0])))
    ended, _ = attack(ending, jnp.int32(SHOOT), ending.battle_key, rolls)
    assert not ended.in_battle and not jnp.any(ended.imp)


def test_witch_script_prefers_weapon_immunity_then_highest_hp_and_mind_blocks(elemental_attack_game):
    from stoix.envs.number_grid import CONTINUE, IMMUNE, WARD
    env, initial, _ = elemental_attack_game
    start = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    start = start.replace(unit_ids=start.unit_ids.at[0].set(env.progression.enemy_ids[0,0]))
    start = compiled_method(env,'_begin_battle')(start.replace(enemy=jnp.int32(7))).replace(actor=jnp.int32(10))
    choose, attack = compiled_method(env,'_enemy_action'), compiled_method(env,'_battle_step')
    assert choose(start, jnp.zeros(6)) == SHOOT  # despite another unit's higher HP
    rolls = jnp.zeros(env.random_size).at[36:].set(.99)
    blocked, _ = attack(start, jnp.int32(CONTINUE), start.battle_key, rolls)
    assert blocked.last_event == IMMUNE and not jnp.any(blocked.imp)
    no_immune = start.replace(hp=start.hp.at[0].set(0))
    assert choose(no_immune, jnp.zeros(6)) == SHOOT+1
    # The same primary source consumes one ward, then transforms on the next hit.
    start = start.replace(unit_ids=start.unit_ids.at[0].set(env.progression.enemy_ids[0,1]))
    warded, _ = attack(start, jnp.int32(CONTINUE), start.battle_key, rolls)
    assert warded.last_event == WARD and not warded.imp[0]
    transformed, _ = attack(warded.replace(actor=jnp.int32(10)), jnp.int32(CONTINUE), start.battle_key, rolls)
    assert transformed.imp[0] and transformed.hp[0] == start.hp[0]
    assert transformed.imp_wards[0] == 1 << 6 and transformed.wards_used[0] == 0


def test_warrior_reference_peasant_melee_tiers_and_script(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import UNITS
    unit = UNITS['peasant']
    assert MAP['enemy_rosters'][8][0] == 'peasant'
    assert tuple(unit[k] for k in ('max_hp', 'damage', 'accuracy', 'armor', 'initiative', 'level')) == (40, 15, 75, 0, 30, 1)
    env, initial, _, attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    # Exercise all four reach tiers, both sides, blocked rear, pending retreat,
    # and escaped front allies. Lower rear HP cannot bypass an occupied front.
    health = jnp.array([[40,30,1,1,1,1], [0,0,20,1,1,1], [0,0,0,10,20,1],
                       [0,0,0,0,0,10], [40,30,1,1,1,1], [40,30,1,1,1,1]], jnp.int32)
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (6,)+a.shape), state)
    states = states.replace(actor=jnp.array([6,6,6,6,9,9], jnp.int32),
        unit_ids=jnp.full((6,12), env.progression.ids['peasant'], jnp.int32),
        hp=states.hp.at[:, :6].set(health),
        retreating=states.retreating.at[0, :3].set(True),
        escaped=states.escaped.at[5, 6:9].set(True))
    choices = compiled_method(env,'_enemy_action',batched=True)(states, jnp.zeros((6,6)))
    chex.assert_trees_all_equal(choices, jnp.array([SHOOT+1, SHOOT+2, SHOOT+3, SHOOT+5, DEFEND, SHOOT+1]))
    mirror = states.replace(actor=states.actor-6,
        hp=jnp.concatenate((states.hp[:,6:],states.hp[:,:6]),axis=1),
        retreating=jnp.concatenate((states.retreating[:,6:],states.retreating[:,:6]),axis=1),
        escaped=jnp.concatenate((states.escaped[:,6:],states.escaped[:,:6]),axis=1))
    masks = compiled_method(env,'action_mask',batched=True)(mirror)[:,SHOOT:SHOOT+6]
    chex.assert_trees_all_equal(masks, jnp.array([[1,1,0,0,0,0], [0,0,1,0,0,0],
        [0,0,0,1,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0], [1,1,0,0,0,0]], jnp.bool_))
    # Accuracy uses two integer draws: mean 74 hits, mean 75 misses.
    sample = jax.tree.map(lambda a:a[0], states)
    sample = sample.replace(unit_levels=jnp.ones(12, jnp.int32))
    hit, _ = attack(sample, jnp.int32(CONTINUE), sample.battle_key, jnp.zeros(env.random_size).at[12:24].set(.74))
    miss, _ = attack(sample, jnp.int32(CONTINUE), sample.battle_key, jnp.zeros(env.random_size).at[12:24].set(.75))
    assert hit.last_target == 1 and hit.last_damage == 15
    assert miss.last_target == 1 and miss.last_damage == 0


def test_centaur_critical_integer_rounding_ignores_secondary_power_armor_and_defend():
    from stoix.envs.number_grid_combat import UNITS, DAMAGE
    profile = UNITS['centaur_savage']
    assert MAP['enemy_rosters'][9][2] == 'centaur_savage'
    assert tuple(profile[k] for k in ('max_hp','damage','accuracy','armor','initiative')) == (210,100,80,0,35)
    config = copy.deepcopy(MAP)
    config['hero_roster'] = ['possessed', 'centaur_savage', 'possessed', 'cultist', 'cultist', None]
    config['hero_units'] = 5
    config['hero_combat_stats'] = [{}, dict(secondary_attack_type='mind', secondary_accuracy=0, secondary_damage=999), {}, {}, {}]
    overrides = [[{} for _ in range(n)] for n in config['enemy_units']]
    overrides[11][0] = dict(max_hp=200, armor=90, immunities=['mind'], protections=['mind'])
    overrides[11][1] = dict(protections=['weapon'])
    overrides[11][2] = dict(immunities=['weapon'])
    config['enemy_combat_stats'] = overrides
    env = NumberGrid(map_config={**config,'initial_potions':{},'chests':[]})
    initial, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(1))
    start = start.replace(hp=start.hp.at[6].set(100), defended=start.defended.at[6].set(True))
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (7,)+a.shape), start)
    states = states.replace(hp=states.hp.at[4,6].set(3),
        unit_levels=states.unit_levels.at[5,1].set(4), imp=states.imp.at[6,1].set(True))
    actions = jnp.array([SHOOT,SHOOT,SHOOT+1,SHOOT+2,SHOOT,SHOOT,SHOOT],jnp.int32)
    rolls = jnp.zeros((7,env.random_size)).at[1,12:24].set(.99).at[:,36:].set(.99)
    result, _ = scalar_battle(env)(states,actions,states.battle_key,rolls)
    # Defend cannot raise 90 armor further: 100 damage removes 10 HP, plus 5 critical.
    # At level 4: 112 - trunc(112*.9) = 12 primary + round(5.6)=6 extra. Imp has no crit.
    chex.assert_trees_all_equal(result.last_damage, jnp.array([15,0,0,0,3,18,2]))
    chex.assert_trees_all_equal(result.hp[:,6], jnp.array([85,100,100,100,0,82,98]))
    assert compiled_method(env,'unit_stats')(jax.tree.map(lambda a:a[5],states))[1,DAMAGE] == 112
    assert result.wards_used[0,6] == 0 and result.wards_used[2,7] == 1
    assert result.last_immune[3] == 1 << 8


@pytest.mark.core
def test_demon_two_strikes_retarget_and_consume_one_activation(current_game):
    from stoix.envs.number_grid import CONTINUE, WAIT, RETREAT
    from stoix.envs.number_grid_combat import UNITS
    assert MAP['enemy_rosters'][10][0] == 'blade_master'
    assert UNITS['blade_master']['damage'] == 75 and UNITS['blade_master']['max_hp'] == 250
    env, initial, step, attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    state = state.replace(unit_ids=state.unit_ids.at[0].set(env.progression.ids['blade_master']),
        unit_levels=state.unit_levels.at[0].set(5), hp=state.hp.at[6:8].set(jnp.array([30,45])))
    rolls = jnp.zeros(env.random_size).at[36:].set(.99)
    first, _ = attack(state, jnp.int32(SHOOT), state.battle_key, rolls)
    assert first.hp[6] == 0 and first.actor == 0 and first.second_strike and first.turn_phase[0] == 0
    mask = compiled_method(env,'action_mask')(first)
    assert mask[SHOOT+1] and not jnp.any(mask[jnp.array([DEFEND,WAIT,RETREAT])])
    invalid, _ = step(first, jnp.int32(WAIT))
    chex.assert_trees_all_equal(invalid.hp, first.hp)
    chex.assert_trees_all_equal(invalid.battle_key, first.battle_key)
    second, _ = attack(first, jnp.int32(SHOOT+1), first.battle_key, rolls)
    assert second.hp[7] == 0 and not second.second_strike and second.turn_phase[0] == 2
    assert second.player_turns == state.player_turns+2
    waited, _ = attack(state, jnp.int32(WAIT), state.battle_key, rolls)
    assert waited.turn_phase[0] == 1 and not waited.second_strike
    resumed = waited.replace(actor=jnp.int32(0))
    waiting_first, _ = attack(resumed, jnp.int32(SHOOT), resumed.battle_key, rolls)
    assert not waiting_first.second_strike and waiting_first.turn_phase[0] == 2
    assert waiting_first.actor != 0 and waiting_first.hp[6] == 0
    guarded, _ = attack(state, jnp.int32(DEFEND), state.battle_key, rolls)
    assert guarded.defended[0] and guarded.turn_phase[0] == 2 and not guarded.second_strike
    imp, _ = attack(state.replace(imp=state.imp.at[0].set(True)), jnp.int32(SHOOT), state.battle_key, rolls)
    assert not imp.second_strike and imp.turn_phase[0] == 2
    # RED chooses a fresh legal lowest-HP target after its first kill.
    red = state.replace(actor=jnp.int32(6),
        unit_ids=state.unit_ids.at[6].set(env.progression.ids['blade_master']),
        unit_levels=state.unit_levels.at[6].set(5), hp=state.hp.at[:3].set(jnp.array([40,30,1])))
    red_first, _ = attack(red, jnp.int32(CONTINUE), red.battle_key, rolls)
    red_second, _ = attack(red_first, jnp.int32(CONTINUE), red_first.battle_key, rolls)
    assert red_first.last_target == 1 and red_second.last_target == 0
    assert red_first.actor == 6 and red_first.second_strike and not red_second.second_strike
    # Public steps stop for the player's second target, but execute both RED hits internally.
    blue, ts = step(state, jnp.int32(SHOOT))
    assert blue.actor == 0 and blue.second_strike and blue.step_count == state.step_count+1
    assert ts.extras['player_battle_transition'] == 1 and ts.extras['automatic_battle_transition'] == 0
    red = red.replace(turn_phase=jnp.full(12, 2, jnp.int32).at[6].set(0).at[3].set(0),
                      priority=jnp.zeros(12).at[6].set(100).at[3].set(90),
                      hp=red.hp.at[:6].set(jnp.where(red.hp[:6] > 0, 1000, 0)))
    after, ts = step(red, jnp.int32(CONTINUE))
    assert after.actor == 3 and not after.second_strike and after.step_count == red.step_count
    assert ts.extras['enemy_battle_transition'] == 2 and ts.extras['automatic_battle_transition'] == 2


def test_poison_queue_wraps_round_and_checks_tick_immunity_once():
    from stoix.envs.number_grid_effects import advance_poison_queue
    hp = jnp.zeros((4,12),jnp.int32).at[:,0].set(5).at[:,1].set(100).at[:,6].set(100)
    phases = jnp.full((4,12),2,jnp.int32).at[:2,0].set(0).at[2:,1].set(0)
    activated = jnp.zeros((4,12),jnp.bool_).at[:2,1].set(True).at[2,1].set(True)
    turns = jnp.zeros((4,12),jnp.int32).at[:,0].set(1).at[:,1].set(2)
    damage = jnp.zeros((4,12),jnp.int32).at[:,0].set(10).at[:3,1].set(15)
    immune = jnp.zeros((4,12),jnp.uint32).at[1,1].set(1 << 4)
    def advance(health, phase, done, left, amount, immunity, actor):
        return advance_poison_queue(health, jnp.zeros(12,jnp.bool_), phase,
            jnp.arange(12,0,-1,dtype=jnp.float32), jnp.int32(1), done,
            left, amount, jnp.full(12,7,jnp.int32), immunity,
            jnp.zeros(12).at[1].set(100.), actor, jnp.bool_(False), 75,jnp.ones(12,jnp.int32),jnp.zeros((2,12)))
    result = jax.jit(jax.vmap(advance))(hp,phases,activated,turns,damage,immune,jnp.array([0,0,1,1]))
    chex.assert_trees_all_equal(result[0][:,1], jnp.array([85,100,100,100]))
    chex.assert_trees_all_equal(result[3], jnp.array([2,2,1,1]))
    chex.assert_trees_all_equal(result[5][:,1], jnp.array([2,0,2,2]))
    chex.assert_trees_all_equal(result[8], jnp.ones(4,jnp.int32))
    assert result[7][1,1] == -1  # poison immunity cancels the cached effect


def test_death_poison_uses_own_accuracy_source_and_multiple_targets(elemental_attack_game):
    from stoix.envs.number_grid_combat import UNITS
    assert MAP['enemy_rosters'][12][0] == 'death'
    death = UNITS['death']
    assert tuple(death[k] for k in ('max_hp','damage','accuracy','initiative','secondary_damage','secondary_accuracy')) == (125,100,80,40,20,50)
    assert set(death['immunities']) == {'weapon','death'}
    env,initial,_ = elemental_attack_game
    initial=hero_roster_state(env,initial,['possessed','duke','possessed','imperial_assassin','cultist',None])
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(36))).replace(actor=jnp.int32(3))
    start = start.replace(priority=start.priority.at[0].set(10000.))
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (7,)+a.shape), start)
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3,SHOOT+4,SHOOT+5,SHOOT+4],jnp.int32)
    rolls = jnp.zeros((7,env.random_size)).at[:,36].set(.99).at[:,37:49].set(.74).at[:,49:55].set(.99)
    rolls = rolls.at[4,12:24].set(.99).at[5,37:49].set(.75)
    result, _ = scalar_attack_cases(env,states,actions,rolls)
    chex.assert_trees_all_equal(result.last_damage,jnp.array([60,60,0,60,0,60,60]))
    chex.assert_trees_all_equal(jnp.sum(result.poison_mode,axis=1),jnp.array([0,0,0,2,0,0,2]))
    assert result.wards_used[0,6] == 1 << 5
    assert result.last_immune[1] == 1 << 7 and result.last_immune[2] == 1 << 8
    assert result.poison_source[3,9] == 3  # source is tracked without a target lock
    cast = compiled_method(env,'_battle_step')
    first = jax.tree.map(lambda a:a[6],result).replace(actor=jnp.int32(3))
    more, _ = cast(first,jnp.int32(SHOOT+5),first.battle_key,rolls[6])
    assert more.poison_mode[10] == 2 and more.poison_mode[11] == 2
    refreshed, _ = cast(first,jnp.int32(SHOOT+4),first.battle_key,rolls[6].at[49:55].set(0.))
    assert refreshed.poison_mode[10] == 2  # reapplication refreshes a long effect without stacking
    # Poison immunity is checked when the unit's activation begins, even though
    # the initial status source was death. No HP tick is applied in that case.
    poisoned = jax.tree.map(lambda a:a[3],result).replace(actor=jnp.int32(0),
        priority=jnp.zeros(12).at[9].set(10000.),
        activation_done=result.activation_done[3].at[9].set(False))
    cleared, _ = cast(poisoned,jnp.int32(DEFEND),poisoned.battle_key,rolls[0])
    assert cleared.hp[9] == 240 and cleared.poison_mode[9] == 0
    secondary = jax.jit(env.progression.secondary_stats)(jnp.int32(env.progression.ids['imperial_assassin']),jnp.int32(4))
    chex.assert_trees_all_equal(secondary,jnp.array([26.,76.]))


def test_wight_secondary_roll_devolution_recovery_and_battle_cleanup():
    from stoix.envs.number_grid import CONTINUE, WAIT, DECAY_TRANSFORMED
    from stoix.envs.number_grid_combat import HP, DAMAGE, MELEE, UNITS
    assert MAP['enemy_rosters'][13][0] == 'wight'
    profile = UNITS['wight']
    assert tuple(profile[k] for k in ('max_hp','damage','accuracy','secondary_accuracy','initiative')) == (105,75,80,80,50)
    config = copy.deepcopy(MAP)
    config['hero_roster'] = ['witch', 'duke', 'possessed', 'wight', 'cultist', None]
    config['hero_units'] = 5
    config['hero_combat_stats'] = [{},{},{},dict(attack_type='fire'),{}]
    config['enemy_rosters'][11] = ['imperial_knight','imperial_knight','orc','imperial_knight','squire','imperial_knight']
    overrides = [[{} for _ in range(n)] for n in config['enemy_units']]
    extras = [dict(protections=['death']),dict(immunities=['death']),dict(protections=['death']),dict(protections=['weapon']),{},{}]
    overrides[11] = [dict(max_hp=300,**e) for e in extras]
    config['enemy_combat_stats'] = overrides
    env = NumberGrid(map_config={**config,'initial_potions':{},'chests':[]})
    initial, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(3))
    start = start.replace(priority=start.priority.at[1].set(10000.),
                          wards_used=start.wards_used.at[9].set(1))
    states = jax.tree.map(lambda a:jnp.broadcast_to(a,(7,)+a.shape),start)
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3,SHOOT+4,SHOOT+5,SHOOT+3],jnp.int32)
    rolls = jnp.zeros((7,env.random_size)).at[:,36].set(.99).at[:,37:49].set(.79)
    rolls = rolls.at[5,37:49].set(.8).at[6,12:24].set(.99)
    out,_ = scalar_attack_cases(env,states,actions,rolls)
    chex.assert_trees_all_equal(out.last_damage,jnp.array([75,75,75,75,75,75,0]))
    chex.assert_trees_all_equal(jnp.sum(out.decay_form>0,axis=1),jnp.array([0,0,0,1,0,0,0]))
    assert out.wards_used[0,6] == 1 << 5 and out.last_immune[1] == 1 << 7
    assert not out.last_immune[2] and out.wards_used[2,8] == 0  # neutral never spends a secondary ward
    first = jax.tree.map(lambda a:a[3],out)
    assert first.last_event == DECAY_TRANSFORMED and first.decay_form[9] == env.progression.ids['knight']
    assert first.hp[9] == 112 and first.wards_used[9] == 0 and first.imp_wards[9] == 1
    assert compiled_method(env,'unit_stats')(first)[9,HP] == 150 and compiled_method(env,'unit_stats')(first)[9,DAMAGE] == 50
    assert env.unit_traits(first)[9,0] == MELEE and first.unit_ids[9] == start.unit_ids[9]
    attack = compiled_method(env,'_battle_step')
    second,_ = attack(first.replace(actor=jnp.int32(3)),jnp.int32(SHOOT+3),first.battle_key,rolls[3])
    assert second.decay_form[9] == env.progression.ids['squire'] and second.hp[9] == 25
    # Drain Level does not recover randomly, including after Wait.
    base = first.replace(actor=jnp.int32(9),activation_done=first.activation_done.at[9].set(False))
    recover = compiled_method(env,'_start_activation')
    restored = recover(base,jnp.float32(.0099))
    failed = recover(base,jnp.float32(.01))
    assert restored.decay_form[9] == base.decay_form[9] and restored.hp[9] == base.hp[9]
    assert failed.decay_form[9] != 0
    waited = recover(base.replace(turn_phase=base.turn_phase.at[9].set(1),priority=base.priority.at[9].set(-1.),
        activation_done=base.activation_done.at[9].set(True)),jnp.float32(0))
    assert waited.decay_form[9] != 0
    # A hostile Witch overlay still uses its own round-age recovery.
    imp,_ = attack(first.replace(actor=jnp.int32(0)),jnp.int32(SHOOT+3),first.battle_key,rolls[3])
    assert imp.imp[9] and imp.decay_form[9] != 0 and compiled_method(env,'unit_stats')(imp)[9,HP] == 150
    nested = imp.replace(actor=jnp.int32(9),activation_done=imp.activation_done.at[9].set(False))
    assert recover(nested,jnp.float32(.1)).imp[9]
    final = recover(nested,jnp.float32(.995))
    assert not final.imp[9] and final.decay_form[9] == 0 and final.hp[9] == 224
    # Battle-end cleanup restores both forms even for a dead unit, without
    # resurrecting it or awarding a spent activation again.
    limited = imp.replace(actor=jnp.int32(1), retreating=imp.retreating.at[1].set(True),
                          escaped=imp.escaped.at[:6].set(True).at[1].set(False))
    ended,_ = attack(limited,jnp.int32(CONTINUE),limited.battle_key,rolls[3])
    assert not ended.in_battle and not jnp.any(ended.decay_form) and not jnp.any(ended.imp)
    assert ended.hp[9] == 224 and ended.unit_levels[9] == start.unit_levels[9]
    corpse = limited.replace(hp=limited.hp.at[9].set(0))
    dead,_ = attack(corpse,jnp.int32(CONTINUE),corpse.battle_key,rolls[3])
    assert dead.hp[9] == 0 and not jnp.any(dead.decay_form)
    # A scripted Wight follows ordinary ranged minimum-HP targeting.
    enemy = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(13))).replace(actor=jnp.int32(6))
    assert env._enemy_action(enemy,jnp.zeros(6)) == SHOOT+4
    assert compiled_method(env,'action_mask')(first.replace(actor=jnp.int32(0)))[WAIT]
    assert compiled_method(env,'action_mask')(enemy)[CONTINUE]


@pytest.mark.core
def test_current_melee_geometry_exhaustive_occupancy_on_both_sides(current_game):
    import numpy as np
    env, initial, _, _ = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    actors, health, expected = [], [], []
    for actor in range(12):
        own, other = (0,6) if actor < 6 else (6,0)
        local = actor % 6
        for enemy_mask in range(64):
            enemies = {i for i in range(6) if enemy_mask & (1 << i)}
            for front_mask in range(8):
                allies = {i for i in range(3) if front_mask & (1 << i)} | {local}
                hp = [0]*12
                for i in allies:
                    hp[own+i] = 100
                for i in enemies:
                    hp[other+i] = 100
                targets = set()
                if local < 3 or not allies & {0,1,2}:
                    near = ({0,1},{0,1,2},{1,2})[local % 3]
                    for row in ({0,1,2},{3,4,5}):
                        candidates = enemies & row
                        adjacent = {i for i in candidates if i % 3 in near}
                        if adjacent or candidates:
                            targets = adjacent or candidates
                            break
                actors.append(actor)
                health.append(hp)
                expected.append([i in targets for i in range(6)])
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(len(actors),)+x.shape),start)
    states = states.replace(actor=jnp.array(actors,jnp.int32),hp=jnp.array(health,jnp.int32))
    actual = jax.jit(jax.vmap(lambda s: jnp.where(s.actor < 6,
        env._melee_targets(s), env._melee_targets(s, enemy_side=True))))(states)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('key,kind,actor,squad,slot,primary,secondary,bit',[
    ('sentry','water',3,16,0,55,20,8),
    ('watcher','burn',4,17,3,60,15,4)])
def test_sentry_and_watcher_secondary_source_accuracy_and_refresh(
        elemental_attack_game,key,kind,actor,squad,slot,primary,secondary,bit):
    from stoix.envs.number_grid_combat import UNITS
    assert MAP['enemy_rosters'][squad][slot] == key
    profile = UNITS[key]
    assert tuple(profile[k] for k in ('max_hp','damage','accuracy','initiative','secondary_damage','secondary_accuracy')) == (120,primary,85,70,secondary,70)
    assert profile['attack_type'] == profile['secondary_attack_type'] == ('fire' if kind == 'burn' else 'water')
    env,initial,start = elemental_attack_game
    start = start.replace(actor=jnp.int32(actor))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(9,)+x.shape),start)
    states = states.replace(unit_levels=states.unit_levels.at[8,actor].set(4),**{
        kind+'_mode':getattr(states,kind+'_mode').at[7,10].set(2),
        kind+'_damage':getattr(states,kind+'_damage').at[7,10].set(7)})
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3,SHOOT+4,SHOOT+5,SHOOT+4,SHOOT+4,SHOOT+5],jnp.int32)
    rolls = jnp.zeros((9,env.random_size)).at[:,36].set(.99).at[:,37:49].set(.69).at[:,49:55].set(.99)
    rolls = rolls.at[4,12:24].set(.99).at[5,37:49].set(.70)
    result,_ = scalar_battle(env)(states,actions,states.battle_key,rolls)
    chex.assert_trees_all_equal(result.last_damage,jnp.array([primary,primary,0,primary,0,primary,primary,primary,primary+6]))
    targets = actions-SHOOT+6
    chex.assert_trees_all_equal(getattr(result,kind+'_mode')[jnp.arange(9),targets],jnp.array([0,0,0,2,0,0,2,2,2]))
    assert result.wards_used[0,6] == bit and result.last_immune[1] == 1 << 7
    assert getattr(result,kind+'_damage')[3,9] == secondary and getattr(result,kind+'_damage')[7,10] == secondary
    assert getattr(result,kind+'_damage')[8,11] == secondary+6
    assert getattr(result,kind+'_source')[3,9] == actor
    first = jax.tree.map(lambda x:x[6],result).replace(actor=jnp.int32(actor))
    second,_ = compiled_method(env,'_battle_step')(first,jnp.int32(SHOOT+3),first.battle_key,rolls[6])
    assert getattr(second,kind+'_mode')[9] == getattr(second,kind+'_mode')[10] == 2
    red = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(squad))).replace(actor=jnp.int32(6+slot))
    red = red.replace(hp=red.hp.at[:5].set(jnp.array([80,50,60,40,70])))
    assert compiled_method(env,'_enemy_action')(red,jnp.zeros(6)) == SHOOT+3


def test_poison_then_water_ticks_skip_dead_units_and_wait_and_award_xp(current_game):
    from stoix.envs.number_grid import WAIT, VICTORY
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(3))
    start = start.replace(hp=start.hp.at[:3].set(jnp.array([10,20,100])),
        priority=jnp.arange(12,0,-1,dtype=jnp.float32),
        activation_done=jnp.zeros(12,jnp.bool_),turn_phase=jnp.zeros(12,jnp.int32),
        poison_mode=jnp.zeros(12,jnp.int32).at[:2].set(2),
        poison_damage=jnp.zeros(12,jnp.int32).at[:2].set(jnp.array([15,5])),
        water_mode=jnp.zeros(12,jnp.int32).at[:3].set(2),
        water_damage=jnp.zeros(12,jnp.int32).at[:2].set(20))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    after,_ = attack(start,jnp.int32(DEFEND),start.battle_key,rolls)
    chex.assert_trees_all_equal(after.hp[:3],jnp.array([0,0,90]))
    chex.assert_trees_all_equal(after.last_poison_damage[:3],jnp.array([10,5,0]))
    chex.assert_trees_all_equal(after.last_water_damage[:3],jnp.array([0,15,10]))
    assert after.actor == 2 and after.water_mode[2] == 2 and after.activation_done[2]
    assert after.battle_xp[0] == compiled_method(env,'unit_experience')(start)[:2,1].sum()
    waiting = after.replace(turn_phase=jnp.full(12,2,jnp.int32).at[2].set(0))
    waited,_ = attack(waiting,jnp.int32(WAIT),waiting.battle_key,rolls)
    assert waited.actor == 2 and waited.hp[2] == 90 and waited.water_mode[2] == 2
    fatal = start.replace(hp=start.hp.at[6:].set(0).at[6].set(10),
        priority=jnp.zeros(12).at[6].set(10000.),
        poison_mode=jnp.zeros(12,jnp.int32),water_mode=jnp.zeros(12,jnp.int32).at[6].set(1),
        water_damage=jnp.zeros(12,jnp.int32).at[6].set(20))
    won,_ = attack(fatal,jnp.int32(DEFEND),fatal.battle_key,rolls)
    assert won.last_event == VICTORY and not won.in_battle
    assert not jnp.any(won.water_mode) and won.last_water_damage[6] == 10
    assert won.battle_xp[1] == compiled_method(env,'unit_experience')(fatal)[6,1]


def test_water_tick_immunity_and_expiry_are_independent_of_poison():
    from stoix.envs.number_grid_effects import advance_periodic_queue
    hp = jnp.zeros(12,jnp.int32).at[0].set(100).at[6].set(100)
    phases = jnp.full(12,2,jnp.int32).at[0].set(0)
    left = jnp.zeros((12,2),jnp.int32).at[0].set(jnp.array([2,1]))
    damage = jnp.zeros((12,2),jnp.int32).at[0].set(jnp.array([15,20]))
    def run(immune):
        return advance_periodic_queue(hp,jnp.zeros(12,jnp.bool_),phases,jnp.arange(12,0,-1,dtype=jnp.float32),
            jnp.int32(1),jnp.zeros(12,jnp.bool_),left,damage,jnp.full((12,2),-1,jnp.int32),
            jnp.zeros(12,jnp.uint32).at[0].set(immune),jnp.ones(12),jnp.int32(0),jnp.bool_(False),75,
            jnp.array([16,8],jnp.uint32),jnp.array([0,10],jnp.int32),jnp.ones((12,2),jnp.int32),jnp.zeros((2,12,2)))
    results = jax.jit(jax.vmap(run))(jnp.array([0,8,16,24],jnp.uint32))
    chex.assert_trees_all_equal(results[0][:,0],jnp.array([65,85,80,100]))
    chex.assert_trees_all_equal(results[5][:,0],jnp.array([[2,0],[2,0],[0,0],[0,0]]))
    assert not jnp.any(results[6][:,0,1])


@pytest.mark.core
def test_poison_water_fire_order_stops_later_effects_after_death(current_game):
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(3))
    start = start.replace(hp=start.hp.at[:3].set(jnp.array([10,20,100])),
        priority=jnp.arange(12,0,-1,dtype=jnp.float32),turn_phase=jnp.zeros(12,jnp.int32),
        activation_done=jnp.zeros(12,jnp.bool_),
        poison_mode=jnp.zeros(12,jnp.int32).at[:3].set(2),
        poison_damage=jnp.zeros(12,jnp.int32).at[:3].set(jnp.array([15,5,5])),
        burn_mode=jnp.zeros(12,jnp.int32).at[:3].set(2),
        burn_damage=jnp.zeros(12,jnp.int32).at[:2].set(20),
        water_mode=jnp.zeros(12,jnp.int32).at[:3].set(2),
        water_damage=jnp.zeros(12,jnp.int32).at[:3].set(20))
    result,_ = attack(start,jnp.int32(DEFEND),start.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    chex.assert_trees_all_equal(result.hp[:3],jnp.array([0,0,65]))
    chex.assert_trees_all_equal(result.last_poison_damage[:3],jnp.array([10,5,5]))
    chex.assert_trees_all_equal(result.last_burn_damage[:3],jnp.array([0,0,10]))
    chex.assert_trees_all_equal(result.last_water_damage[:3],jnp.array([0,15,20]))
    assert result.actor == 2 and result.activation_done[2]
    assert result.battle_xp[0] == compiled_method(env,'unit_experience')(start)[:2,1].sum()


def test_bone_lord_leech_overkill_sharing_defend_miss_and_imp(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import UNITS
    env,initial,_,_ = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    assert MAP['enemy_rosters'][18][0] == 'bone_lord'
    assert UNITS['bone_lord']['max_hp'] == 400 and UNITS['bone_lord']['damage'] == 65
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(18))).replace(actor=jnp.int32(6))
    start = start.replace(hp=start.hp.at[:5].set(jnp.array([300,50,300,300,300])).at[6].set(350).at[7].set(44).at[9].set(30).at[10].set(45),
        priority=jnp.zeros(12).at[4].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(7,)+x.shape),start)
    states = states.replace(hp=states.hp.at[1,6].set(390).at[2,6].set(400).at[6,1].set(1),
        defended=states.defended.at[3,1].set(True),imp=states.imp.at[5,6].set(True))
    rolls = jnp.zeros((7,env.random_size)).at[:,36].set(.99).at[4,12:24].set(.99)
    result,_ = scalar_battle(env)(states,jnp.full(7,CONTINUE,jnp.int32),states.battle_key,rolls)
    chex.assert_trees_all_equal(result.last_damage,jnp.array([50,50,50,33,0,20,1]))
    chex.assert_trees_all_equal(result.hp[:,6],jnp.array([375,400,400,366,350,350,350]))
    chex.assert_trees_all_equal(result.hp[:,jnp.array([7,9,10])],jnp.array([
        [44,30,45],[45,39,50],[45,45,50],[44,30,45],[44,30,45],[44,30,45],[44,30,45]]))


def test_vampiric_sharing_matches_reference_integer_distribution():
    import numpy as np
    from stoix.envs.number_grid_effects import vampiric_heal
    rng = np.random.default_rng(42)
    maximum = rng.integers(1,1000,(1024,12),dtype=np.int32)
    hp = np.minimum(rng.integers(0,1000,(1024,12),dtype=np.int32),maximum)
    actors = rng.integers(0,12,1024,dtype=np.int32)
    hp[np.arange(1024),actors] = np.maximum(hp[np.arange(1024),actors],1)
    escaped = rng.random((1024,12)) < .2
    escaped[np.arange(1024),actors] = False
    pools = rng.integers(0,1501,1024,dtype=np.int32)
    sharing = rng.random(1024) < .8
    expected = hp.copy()
    for n in range(1024):
        actor = actors[n]
        heal = min(pools[n],maximum[n,actor]-expected[n,actor])
        expected[n,actor] += heal
        left = pools[n]-heal if sharing[n] else 0
        while left > 0:
            targets = [i for i in range(12) if i != actor and (i < 6) == (actor < 6)
                and not escaped[n,i] and 0 < expected[n,i] < maximum[n,i]]
            if not targets:
                break
            share = max(1,left//len(targets))
            for i in targets:
                heal = min(share,maximum[n,i]-expected[n,i],left)
                expected[n,i] += heal
                left -= heal
    actual = jax.jit(jax.vmap(vampiric_heal))(*map(jnp.asarray,(hp,maximum,escaped,actors,pools,sharing)))
    np.testing.assert_array_equal(actual,expected)


def test_dead_dragon_area_poison_has_independent_hits_and_no_caster_lock(current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_,attack = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    assert MAP['enemy_rosters'][21][0] == 'dead_dragon'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(21))).replace(actor=jnp.int32(6))
    start = start.replace(hp=start.hp.at[:5].set(jnp.array([10,500,500,500,500])),
        priority=jnp.zeros(12).at[7].set(10000.),escaped=start.escaped.at[3].set(True))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[13].set(.99).at[19].set(.99).at[41].set(.99).at[47].set(.99).at[49:55].set(.99)
    result,_ = attack(start,jnp.int32(CONTINUE),start.battle_key,rolls)
    chex.assert_trees_all_equal(result.hp[:5],jnp.array([0,500,435,500,435]))
    chex.assert_trees_all_equal(result.poison_mode[:5],jnp.array([0,0,2,0,0]))
    assert result.poison_damage[2] == 20 and result.poison_source[2] == 6
    again = result.replace(actor=jnp.int32(6),hp=result.hp.at[1].set(500))
    second,_ = attack(again,jnp.int32(CONTINUE),again.battle_key,rolls.at[13].set(0).at[19].set(0))
    assert second.poison_mode[1] == 2 and second.poison_mode[2] == 2


def test_gumtic_area_fire_has_independent_hits_and_no_caster_lock(current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_,attack = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    assert MAP['enemy_rosters'][21][2] == 'gumtic'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(21))).replace(actor=jnp.int32(8))
    start = start.replace(hp=start.hp.at[:5].set(jnp.array([10,500,500,500,500])),
        priority=jnp.zeros(12).at[7].set(10000.),escaped=start.escaped.at[3].set(True))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[13].set(.99).at[19].set(.99).at[41].set(.99).at[47].set(.99).at[49:55].set(.99)
    result,_ = attack(start,jnp.int32(CONTINUE),start.battle_key,rolls)
    chex.assert_trees_all_equal(result.hp[:5],jnp.array([0,500,390,500,390]))
    chex.assert_trees_all_equal(result.burn_mode[:5],jnp.array([0,0,2,0,0]))
    assert result.burn_damage[2] == 15 and result.burn_source[2] == 8
    again = result.replace(actor=jnp.int32(8),hp=result.hp.at[1].set(500))
    second,_ = attack(again,jnp.int32(CONTINUE),again.battle_key,rolls.at[13].set(0).at[19].set(0))
    assert second.burn_mode[1] == 2 and second.burn_mode[2] == 2


def test_ghost_primary_paralysis_and_gast_long_effect(elemental_attack_game):
    from stoix.envs.number_grid import PARALYZED
    env,initial,start = elemental_attack_game
    assert MAP['enemy_rosters'][20][0] == 'ghost'
    uid = env.progression.ids['ghost']
    start = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(uid),
        unit_levels=start.unit_levels.at[3].set(1))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(7,)+x.shape),start)
    states = states.replace(unit_ids=states.unit_ids.at[5,3].set(env.progression.ids['dark_elf_gast']))
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3,SHOOT+4,SHOOT+5,SHOOT+5])
    rolls = jnp.zeros((7,env.random_size)).at[:,12:24].set(.64).at[:,36].set(.99).at[4,12:24].set(.65)
    out,_ = scalar_battle(env)(states,actions,states.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp,states.hp)
    targets = actions-SHOOT+6
    chex.assert_trees_all_equal(out.paralyzed[jnp.arange(7),targets],jnp.array([0,0,1,1,0,0,1],bool))
    chex.assert_trees_all_equal(out.long_paralyzed[jnp.arange(7),targets],jnp.array([0,0,0,0,0,1,0],bool))
    assert out.wards_used[0,6] == 64 and out.last_immune[1] == 1 << 7
    assert out.last_event[3] == PARALYZED and not jnp.any(out.last_damage)
    red = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(20))).replace(actor=jnp.int32(6))
    red = red.replace(hp=red.hp.at[:5].set(jnp.array([100,50,60,40,70])))
    assert compiled_method(env,'_enemy_action')(red,jnp.zeros(6)) == SHOOT
    red = red.replace(unit_ids=red.unit_ids.at[2].set(env.progression.ids['wight']))
    assert compiled_method(env,'_enemy_action')(red,jnp.zeros(6)) == SHOOT+2


def test_paralysis_periodic_tick_precedes_skip_and_terminal_cleanup(current_game):
    from stoix.envs.number_grid import CONTINUE, DEFEND
    env,initial,_,_ = current_game
    base = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = base.replace(actor=jnp.int32(1),priority=jnp.zeros(12).at[0].set(10000.),
        hp=base.hp.at[0].set(20),paralyzed=base.paralyzed.at[0].set(True),
        activation_done=jnp.zeros(12,bool),poison_mode=base.poison_mode.at[0].set(1),
        poison_damage=base.poison_damage.at[0].set(7))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[67].set(.99)
    attack = compiled_method(env,'_battle_step')
    ticked,_ = attack(state,jnp.int32(DEFEND),state.battle_key,rolls)
    assert ticked.actor == 0 and ticked.hp[0] == 13 and ticked.paralyzed[0]
    skipped,_ = attack(ticked,jnp.int32(CONTINUE),state.battle_key,rolls)
    assert skipped.hp[0] == 13 and not skipped.paralyzed[0]
    victory = base.replace(actor=jnp.int32(3),hp=base.hp.at[6:].set(0).at[6].set(1),
        paralyzed=base.paralyzed.at[0].set(True),long_paralyzed=base.long_paralyzed.at[1].set(True))
    withdrawal = base.replace(actor=jnp.int32(0),hp=base.hp.at[:6].set(0).at[0].set(20),
        retreating=base.retreating.at[0].set(True),paralyzed=base.paralyzed.at[7].set(True))
    loss = base.replace(actor=jnp.int32(6),hp=base.hp.at[:6].set(0).at[0].set(1),
        long_paralyzed=base.long_paralyzed.at[7].set(True))
    timeout = base.replace(actor=jnp.int32(3),step_count=jnp.int32(env.max_steps-1),
        paralyzed=base.paralyzed.at[0].set(True),long_paralyzed=base.long_paralyzed.at[1].set(True))
    states = jax.tree.map(lambda *xs:jnp.stack(xs),victory,withdrawal,loss,timeout)
    out,_ = scalar_battle(env)(states,jnp.array([SHOOT,CONTINUE,CONTINUE,DEFEND]),
        states.battle_key,jnp.broadcast_to(rolls,(4,env.random_size)))
    assert not jnp.any(out.paralyzed[:3]) and not jnp.any(out.long_paralyzed[:3])
    assert not out.in_battle[0] and not out.in_battle[1] and out.lost[2]
    # A raw activation does not end a battle at the decision budget; effects remain for bootstrap.
    assert not out.done[3] and out.in_battle[3] and jnp.any(out.long_paralyzed[3])


def test_shadow_independent_area_paralysis_preserves_wards_on_existing_effect(elemental_attack_game):
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][22][3] == 'shadow'
    start = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(env.progression.ids['shadow']),
        unit_levels=start.unit_levels.at[3].set(env.progression.base_levels[env.progression.ids['shadow']]),
        paralyzed=start.paralyzed.at[6].set(True))
    rolls = jnp.zeros(env.random_size).at[12:24].set(.49).at[16].set(.50).at[22].set(.50).at[36].set(.99)
    out,_ = compiled_method(env,'_battle_step')(start,jnp.int32(SHOOT),start.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp,start.hp)
    chex.assert_trees_all_equal(out.paralyzed[6:],jnp.array([1,0,1,1,0,1],bool))
    assert out.wards_used[6] == 0 and out.last_damage == 0 and out.last_target == -1


def test_succub_area_transform_primary_rolls_secondary_source_and_guard_exclusion(elemental_attack_game):
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][23][5] == 'succubus'
    start = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(env.progression.ids['succubus']),
        unit_levels=start.unit_levels.at[3].set(4))
    rolls = jnp.zeros(env.random_size).at[12:24].set(.39).at[16].set(.40).at[22].set(.40).at[36].set(.99)
    out,_ = compiled_method(env,'_battle_step')(start,jnp.int32(SHOOT),start.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp,start.hp)
    chex.assert_trees_all_equal(out.imp[6:],jnp.array([0,0,1,1,0,1],bool))
    assert out.wards_used[6] == 64 and out.last_damage == 0 and out.last_target == -1


def test_succub_uses_effect_source_and_excludes_guard_before_ward(elemental_attack_game):
    env,initial,_ = elemental_attack_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(40))).replace(actor=jnp.int32(3))
    start = start.replace(priority=jnp.zeros(12).at[0].set(10000.),unit_levels=start.unit_levels.at[3].set(4))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),start)
    states = states.replace(unit_ids=states.unit_ids.at[:,3].set(env.progression.enemy_ids[40,:2]))
    rolls = jnp.zeros((2,env.random_size)).at[:,36].set(.99)
    out,_ = scalar_battle(env)(states,jnp.full(2,SHOOT),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.imp[:,6:],jnp.array([[1,1,0,1,0,0],[1,1,0,0,1,1]],bool))
    chex.assert_trees_all_equal(out.hp,states.hp)
    assert jnp.all(out.wards_used[:,8] == 0) and out.wards_used[0,10] == 64


def test_betrezen_independent_primary_and_secondary_sources(elemental_attack_game):
    env,initial,_ = elemental_attack_game
    assert MAP['enemy_rosters'][27][0] == 'uter_betrezen'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(22))).replace(actor=jnp.int32(3))
    start = start.replace(unit_ids=start.unit_ids.at[3].set(env.progression.ids['uter_betrezen']),
        unit_levels=start.unit_levels.at[3].set(1),priority=jnp.zeros(12).at[0].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(8,)+x.shape),start)
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3,SHOOT+4,SHOOT+5,SHOOT+4,SHOOT])
    states = states.replace(long_paralyzed=states.long_paralyzed.at[7,6].set(True))
    rolls = jnp.zeros((8,env.random_size)).at[:,36].set(.99).at[:,67].set(.99)
    rolls = rolls.at[5,37:49].set(.90).at[6,12:24].set(.90)
    out,_ = scalar_battle(env)(states,actions,states.battle_key,rolls)
    chex.assert_trees_all_equal(out.last_damage,jnp.array([100,100,0,0,100,100,0,100]))
    targets = actions-SHOOT+6
    chex.assert_trees_all_equal(out.long_paralyzed[jnp.arange(8),targets],jnp.array([0,0,0,0,1,0,0,1],bool))
    assert out.wards_used[0,6] == 64 and out.wards_used[2,8] == 4
    assert out.wards_used[7,6] == 0  # existing long paralysis rejects before secondary wards
    assert not jnp.any(out.paralyzed)


def test_betrezen_script_prefers_raw_damage_non_paralyzed_with_low_hp_fallback(current_game):
    env,initial,_,_ = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(27))).replace(actor=jnp.int32(6))
    start = start.replace(hp=start.hp.at[:5].set(jnp.array([100,30,50,40,10])),
        unit_ids=start.unit_ids.at[4].set(env.progression.ids['lord']),
        unit_levels=start.unit_levels.at[4].set(5))
    choose = compiled_method(env,'_enemy_action')
    assert choose(start,jnp.zeros(6)) == SHOOT+4
    assert choose(start.replace(paralyzed=start.paralyzed.at[4].set(True)),jnp.zeros(6)) == SHOOT+1
    assert choose(start.replace(long_paralyzed=start.long_paralyzed.at[:5].set(True)),jnp.zeros(6)) == SHOOT+4


def test_betrezen_empty_secondary_is_untyped_but_zero_accuracy_disables(elemental_attack_game):
    env,initial,_ = elemental_attack_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(22))).replace(actor=jnp.int32(3))
    start = start.replace(priority=jnp.zeros(12).at[0].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),start)
    states = states.replace(unit_ids=states.unit_ids.at[:,3].set(env.progression.enemy_ids[39,:2]),
        unit_levels=states.unit_levels.at[:,3].set(1))
    rolls = jnp.zeros((2,env.random_size)).at[:,36].set(.99)
    out,_ = scalar_battle(env)(states,jnp.full(2,SHOOT+1),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.last_damage,jnp.full(2,100))
    # An empty reference source is untyped: it applies even through mind immunity.
    # Zero secondary accuracy disables the effect before rolling at all.
    chex.assert_trees_all_equal(out.long_paralyzed[:,7],jnp.array([1,0],bool))
    assert not jnp.any(out.paralyzed) and not jnp.any(out.wards_used)


def test_uter_melee_secondary_paralysis_only_after_primary_survivor(elemental_attack_game, current_game):
    env,initial,_ = elemental_attack_game
    assert MAP['enemy_rosters'][29][0] == 'ghoul'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(22))).replace(actor=jnp.int32(1))
    start = start.replace(unit_ids=start.unit_ids.at[1].set(env.progression.ids['ghoul']),
        unit_levels=start.unit_levels.at[1].set(1),priority=jnp.zeros(12).at[0].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(5,)+x.shape),start)
    states = states.replace(hp=states.hp.at[3,8].set(1))
    actions = jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+2,SHOOT+2])
    rolls = jnp.zeros((5,env.random_size)).at[:,36].set(.99).at[:,67].set(.99).at[4,12:24].set(.75)
    out,_ = scalar_battle(env)(states,actions,states.battle_key,rolls)
    chex.assert_trees_all_equal(out.last_damage,jnp.array([35,35,35,1,0]))
    targets = actions-SHOOT+6
    chex.assert_trees_all_equal(out.long_paralyzed[jnp.arange(5),targets],jnp.array([0,0,1,0,0],bool))
    assert out.wards_used[0,6] == 64 and not jnp.any(out.paralyzed)
    # A rear Uter remains a melee fighter; the mask and step deny attacks while its front lives.
    # Reach validation needs no synthetic ward profiles; reuse the public step
    # already compiled for the current map, retaining the same rear/front case.
    shared, initial, advance, _ = current_game
    rear = hero_roster_state(shared,initial,['possessed','duke','possessed','cultist','ghoul',None])
    rear = shared._begin_battle(rear.replace(enemy=jnp.int32(22))).replace(actor=jnp.int32(4))
    assert not jnp.any(compiled_method(shared,'action_mask')(rear)[SHOOT:SHOOT+6])
    rejected,_ = advance(rear,jnp.int32(SHOOT))
    chex.assert_trees_all_equal(rejected.hp,rear.hp)


def test_uter_demon_area_secondary_finite_mermaid_and_long_demon(elemental_attack_game):
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][29][1] == 'mermaid'
    start = start.replace(actor=jnp.int32(2),priority=jnp.zeros(12).at[0].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),start)
    states = states.replace(unit_ids=states.unit_ids.at[:,2].set(jnp.array([env.progression.ids['mermaid'],env.progression.ids['uter_demon']])),
        unit_levels=states.unit_levels.at[:,2].set(1))
    rolls = jnp.zeros((2,env.random_size)).at[:,36].set(.99).at[:,67].set(.99)
    rolls = rolls.at[0,16].set(.60).at[0,22].set(.60).at[1,16].set(.90).at[1,22].set(.90)
    rolls = rolls.at[:,42].set(.99).at[:,48].set(.99)
    out,_ = scalar_battle(env)(states,jnp.full(2,SHOOT),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp[:,6:],jnp.array([[300,300,280,280,300,280],[300,300,150,150,300,150]]))
    chex.assert_trees_all_equal(out.paralyzed[0,6:],jnp.array([0,0,1,1,0,0],bool))
    chex.assert_trees_all_equal(out.long_paralyzed[1,6:],jnp.array([0,0,1,1,0,0],bool))
    assert not jnp.any(out.paralyzed[1]) and not jnp.any(out.long_paralyzed[0])
    assert jnp.all(out.last_target == -1)


def test_tiamat_area_weaken_has_independent_source_accuracy_and_no_stack(elemental_attack_game):
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][32][1] == 'tiamat'
    start = start.replace(actor=jnp.int32(1),unit_ids=start.unit_ids.at[1].set(env.progression.ids['tiamat']),
        unit_levels=start.unit_levels.at[1].set(5),priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[41].set(.80).at[47].set(.80)
    out,_ = compiled_method(env,'_battle_step')(start,jnp.int32(SHOOT),start.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp[6:],jnp.array([200,200,300,200,200,200]))
    chex.assert_trees_all_equal(out.weakened[6:],jnp.array([0,0,0,1,0,1],bool))
    assert out.wards_used[6] == 64
    before = compiled_method(env,'unit_stats')(out)
    repeated,_ = compiled_method(env,'_battle_step')(out.replace(actor=jnp.int32(1)),jnp.int32(SHOOT),out.battle_key,rolls)
    after = compiled_method(env,'unit_stats')(repeated)
    chex.assert_trees_all_equal(before[jnp.array([9,11]),1],after[jnp.array([9,11]),1])


def test_tiamat_modifier_precedes_damage_cap_and_survives_temporary_form(current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_,_ = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    grown = start.replace(unit_levels=start.unit_levels.at[1].set(80),weakened=start.weakened.at[1].set(True))
    # Duke: 162 at level 10, +5 for 70 later levels = 512; round(512*.67)=343.
    assert compiled_method(env,'unit_stats')(grown)[1,1] == 343
    cast = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(env.progression.ids['witch']),
        unit_levels=start.unit_levels.at[3].set(3),weakened=start.weakened.at[6].set(True),
        priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    transformed,_ = compiled_method(env,'_battle_step')(cast,jnp.int32(SHOOT),cast.battle_key,rolls)
    assert transformed.imp[6] and not transformed.weakened[6] and transformed.saved_weakened[6]
    assert compiled_method(env,'unit_stats')(transformed)[6,1] == 20
    recovered = compiled_method(env,'_start_activation')(transformed.replace(actor=jnp.int32(6),
        activation_done=transformed.activation_done.at[6].set(False)),jnp.float32(.995))
    assert not recovered.imp[6] and recovered.weakened[6]
    fleeing = grown.replace(actor=jnp.int32(3),retreating=grown.retreating.at[3].set(True),
                            escaped=grown.escaped.at[:6].set(True).at[3].set(False))
    terminal,_ = compiled_method(env,'_battle_step')(fleeing,jnp.int32(CONTINUE),grown.battle_key,rolls)
    assert not jnp.any(terminal.weakened) and not jnp.any(terminal.saved_weakened)


@pytest.mark.core
def test_baroness_enemy_escape_victory_awards_only_actual_damage_no_kill_xp(current_game):
    from stoix.envs.number_grid import CONTINUE, VICTORY
    env,initial,_,_ = current_game
    assert MAP['enemy_rosters'][30][3] == 'baroness'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(30))).replace(actor=jnp.int32(3))
    start = start.replace(unit_ids=start.unit_ids.at[3].set(env.progression.ids['baroness']),
        unit_levels=start.unit_levels.at[3].set(1),hp=start.hp.at[6:].set(0).at[9].set(100),
        enemy_initial_hp=jnp.zeros(6,jnp.int32).at[3].set(100),priority=jnp.zeros(12).at[9].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),start)
    states = states.replace(hp=states.hp.at[1,9].set(50),enemy_damage_credit=states.enemy_damage_credit.at[1,3].set(50))
    rolls = jnp.zeros((2,env.random_size)).at[:,36].set(.99).at[:,67].set(.99)
    fight = scalar_battle(env)
    feared,_ = fight(states,jnp.full(2,SHOOT+3),states.battle_key,rolls)
    assert jnp.all(feared.feared[:,9]) and jnp.all(feared.actor == 9) and not jnp.any(feared.last_damage)
    won,_ = fight(feared,jnp.full(2,CONTINUE),feared.battle_key,rolls)
    assert jnp.all(won.last_event == VICTORY) and not jnp.any(won.in_battle)
    chex.assert_trees_all_equal(won.recovery_balance,jnp.array([[0,0],[50,0]]))
    # No kills: eligible survivors receive only the original minimum of 1 XP.
    assert jnp.all(won.last_xp <= 1) and not jnp.any(won.battle_xp) and not jnp.any(won.feared)


def test_baroness_highest_hp_ai_and_fear_delayed_by_paralysis(current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_,_ = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(30))).replace(actor=jnp.int32(9))
    start = start.replace(hp=start.hp.at[:5].set(jnp.array([100,150,120,500,40])))
    assert compiled_method(env,'_enemy_action')(start,jnp.zeros(6)) == SHOOT+3
    fleeing = start.replace(actor=jnp.int32(0),feared=start.feared.at[0].set(True),
        retreating=start.retreating.at[0].set(True),paralyzed=start.paralyzed.at[0].set(True),
        priority=jnp.zeros(12).at[1].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[67].set(.99)
    skipped,_ = compiled_method(env,'_battle_step')(fleeing,jnp.int32(CONTINUE),fleeing.battle_key,rolls)
    assert skipped.feared[0] and skipped.retreating[0] and not skipped.escaped[0] and not skipped.paralyzed[0]
    escaped,_ = compiled_method(env,'_battle_step')(skipped.replace(actor=jnp.int32(0)),
        jnp.int32(CONTINUE),skipped.battle_key,rolls)
    assert escaped.escaped[0] and not escaped.feared[0]


def test_baroness_interior_fear_is_finite_paralysis_after_source_checks(elemental_attack_game):
    env,_,start = elemental_attack_game
    start = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(env.progression.ids['baroness']),
        unit_levels=start.unit_levels.at[3].set(1))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(4,)+x.shape),start)
    rolls = jnp.zeros((4,env.random_size)).at[:,36].set(.99).at[:,67].set(.99).at[3,12:24].set(.8)
    out,_ = scalar_battle(env)(states,jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+3]),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp,states.hp)
    chex.assert_trees_all_equal(out.paralyzed[jnp.arange(4),jnp.arange(4)+6],jnp.array([0,0,1,0],bool))
    assert not jnp.any(out.feared) and not jnp.any(out.retreating) and out.wards_used[0,6] == 64


def test_baroness_escape_keeps_temporary_form_until_exit_and_ticks_first(current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(3))
    start = start.replace(imp=start.imp.at[6].set(True),feared=start.feared.at[6].set(True),
        retreating=start.retreating.at[6].set(True),hp=start.hp.at[6].set(50),
        poison_mode=start.poison_mode.at[6].set(2),poison_damage=start.poison_damage.at[6].set(7),
        priority=jnp.zeros(12).at[6].set(10000.),activation_done=jnp.zeros(12,bool))
    rolls = jnp.zeros(env.random_size)
    ready,_ = attack(start,jnp.int32(DEFEND),start.battle_key,rolls)
    assert ready.actor == 6 and ready.hp[6] == 43 and ready.last_poison_damage[6] == 7
    assert ready.imp[6]  # fear escape precedes the otherwise successful recovery roll
    fled,_ = attack(ready,jnp.int32(CONTINUE),ready.battle_key,rolls.at[36].set(.99))
    assert fled.escaped[6] and fled.hp[6] == 43 and not fled.feared[6]


def test_baroness_recovery_credit_caps_damage_and_counts_each_death_once(current_game):
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(3))
    start = start.replace(unit_ids=start.unit_ids.at[3].set(env.progression.ids['archer']),
        unit_levels=start.unit_levels.at[3].set(1),hp=start.hp.at[6].set(40),
        enemy_initial_hp=start.enemy_initial_hp.at[0].set(40),
        priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    one,_ = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    assert one.hp[6] == 15 and one.enemy_damage_credit[0] == 25
    # The recipient is healed back up; a new hit must not buy extra credit.
    healed = one.replace(actor=jnp.int32(3),hp=one.hp.at[6].set(40))
    two,_ = attack(healed,jnp.int32(SHOOT),healed.battle_key,rolls)
    assert two.hp[6] == 15 and two.enemy_damage_credit[0] == 40
    dead,_ = attack(two.replace(actor=jnp.int32(3)),jnp.int32(SHOOT),two.battle_key,rolls)
    assert dead.hp[6] == 0 and dead.enemy_killed_credit[0]
    revived = dead.replace(actor=jnp.int32(3),hp=dead.hp.at[6].set(1))
    dead_again,_ = attack(revived,jnp.int32(SHOOT),revived.battle_key,rolls)
    assert dead_again.enemy_damage_credit[0] == 40 and dead_again.enemy_killed_credit.sum() == 1
    assert dead_again.battle_xp[1] == 2*compiled_method(env,'unit_experience')(start)[6,1]


def test_baroness_hero_growth_keeps_zero_damage_and_reference_milestones(current_game):
    env,initial,_,_ = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(30)))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(5,)+x.shape),start)
    states = states.replace(unit_levels=states.unit_levels.at[:,9].set(jnp.array([1,4,13,14,15])))
    stats = compiled_method(env,'unit_stats',batched=True)(states)
    chex.assert_trees_all_equal(stats[:,9],jnp.array([
        [100,0,80,0,20],[156,0,83,0,20],[231,0,89,0,30],[236,0,100,0,30],[241,0,100,20,30]],jnp.float32))
    xp = compiled_method(env,'unit_experience',batched=True)(states)
    chex.assert_trees_all_equal(xp[:,9,2],jnp.array([120,1020,3720,4020,4320]))


def test_incub_independent_area_paralysis_preserves_wards_on_existing_effect(elemental_attack_game):
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][31][4] == 'medusa'
    start = start.replace(actor=jnp.int32(3),unit_ids=start.unit_ids.at[3].set(env.progression.ids['medusa']),
        unit_levels=start.unit_levels.at[3].set(env.progression.base_levels[env.progression.ids['medusa']]),
        paralyzed=start.paralyzed.at[6].set(True))
    rolls = jnp.zeros(env.random_size).at[12:24].set(.59).at[16].set(.60).at[22].set(.60).at[36].set(.99)
    out,_ = compiled_method(env,'_battle_step')(start,jnp.int32(SHOOT),start.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp,start.hp)
    chex.assert_trees_all_equal(out.paralyzed[6:],jnp.array([1,0,1,1,0,1],bool))
    assert out.wards_used[6] == 0 and out.last_damage == 0 and out.last_target == -1


def test_incub_building_unlocks_demonologist_promotion_and_full_heal(current_game):
    env,initial,_,_ = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    demonologist = env.progression.ids['demonologist']
    slot = next(i for i,row in enumerate(env.construction.rows) if row['unit'] == 'Инкуб')
    start = start.replace(unit_ids=start.unit_ids.at[3].set(demonologist),
        unit_levels=start.unit_levels.at[3].set(3),unit_xp=start.unit_xp.at[3].set(1324),
        hp=start.hp.at[3].set(10).at[6].set(1))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),start)
    states = states.replace(buildings=states.buildings.at[1].set(jnp.uint32(1 << slot)))
    rolls = jnp.zeros((2,env.random_size))
    out,_ = scalar_battle(env)(states,jnp.full(2,SHOOT),states.battle_key,rolls)
    assert out.unit_ids[0,3] == demonologist and out.unit_xp[0,3] == 1324 and out.hp[0,3] == 10
    assert out.unit_ids[1,3] == env.progression.ids['incubus'] and out.unit_xp[1,3] == 0 and out.hp[1,3] == 135


def test_abyss_devil_finite_secondary_paralysis_and_ordinary_melee_script(elemental_attack_game,current_game):
    from stoix.envs.number_grid import CONTINUE
    env,initial,_ = elemental_attack_game
    assert MAP['enemy_rosters'][5][1] == 'abyss_devil' and MAP['enemy_rosters'][5][4] is None
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(22))).replace(actor=jnp.int32(1))
    start = start.replace(unit_ids=start.unit_ids.at[1].set(env.progression.ids['abyss_devil']),
        unit_levels=start.unit_levels.at[1].set(5),priority=jnp.zeros(12).at[0].set(10000.))
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(4,)+x.shape),start)
    rolls = jnp.zeros((4,env.random_size)).at[:,36].set(.99).at[:,67].set(.99).at[3,37:49].set(.50)
    out,_ = scalar_battle(env)(states,jnp.array([SHOOT,SHOOT+1,SHOOT+2,SHOOT+2]),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.last_damage,jnp.full(4,140))
    chex.assert_trees_all_equal(out.paralyzed[jnp.arange(4),jnp.array([6,7,8,8])],jnp.array([0,0,1,0],bool))
    assert not jnp.any(out.long_paralyzed)
    actual,initial,_,_ = current_game
    red = actual._begin_battle(initial.replace(enemy=jnp.int32(5))).replace(actor=jnp.int32(7))
    red = red.replace(hp=red.hp.at[:3].set(jnp.array([100,30,80])),
        unit_ids=red.unit_ids.at[0].set(actual.progression.ids['duke']),unit_levels=red.unit_levels.at[0].set(80))
    assert compiled_method(actual,'_enemy_action')(red,jnp.zeros(6)) == SHOOT+1
    out,_ = compiled_method(actual,'_battle_step')(red,jnp.int32(CONTINUE),red.battle_key,jnp.zeros(actual.random_size))
    assert out.last_target == 1


def test_cliric_minimum_unit_point_heal_and_enemy_script(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import UNITS
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    profile = UNITS['medium']
    assert MAP['enemy_rosters'][33][3] == 'medium'
    assert tuple(profile[k] for k in ('max_hp','damage','initiative','exp_kill','exp_required')) == (45,25,10,25,90)
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','medium','cultist',None])
    state = state.replace(actor=jnp.int32(3),hp=state.hp.at[0].set(60),
        priority=jnp.zeros(12).at[0].set(10000.),defended=state.defended.at[0].set(True))
    rolls = jnp.full(env.random_size,.99)
    healed,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    assert healed.hp[0] == 85 and healed.last_damage == 25
    own,_ = attack(state.replace(hp=state.hp.at[3].set(5)),jnp.int32(SHOOT+3),state.battle_key,rolls)
    assert own.hp[3] == 30
    red = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(33))).replace(actor=jnp.int32(9))
    red = enemy_roster_state(env,red,['squire','archer','archer','medium',None,None])
    red = red.replace(hp=red.hp.at[6].set(90).at[7].set(40))
    assert compiled_method(env,'_enemy_action')(red,jnp.zeros(6)) == SHOOT+1
    result,_ = attack(red,jnp.int32(CONTINUE),red.battle_key,rolls)
    assert result.hp[7] == 45


@pytest.mark.core
def test_cliric_post_victory_restores_forms_and_heals_before_xp(current_game):
    from stoix.envs.number_grid import WAIT, VICTORY, POST_HEAL
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['archer','duke','possessed','medium','priest',None])
    start = start.replace(actor=jnp.int32(0),hp=start.hp.at[0].set(5).at[6:].set(0).at[6].set(1),
        imp=start.imp.at[3].set(True),saved_weakened=start.saved_weakened.at[3].set(True),
        paralyzed=start.paralyzed.at[3].set(True),retreating=start.retreating.at[3].set(True))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    first,reward = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    assert first.in_battle and not first.done and first.actor == 3 and first.post_victory == 1
    assert first.last_event == POST_HEAL and reward == 0 and not jnp.any(first.last_xp)
    assert not first.imp[3] and first.weakened[3] and first.paralyzed[3]
    mask = compiled_method(env,'action_mask')(first)
    assert mask[WAIT] and mask[SHOOT] and not mask[DEFEND]
    assert compiled_method(env,'observation')(first).shape == (1843,)
    bad,_ = compiled_method(env,'step')(first,jnp.int32(DEFEND))
    chex.assert_trees_all_equal(bad.hp,first.hp)
    chex.assert_trees_all_equal(bad.battle_key,first.battle_key)
    healed,_ = attack(first,jnp.int32(SHOOT),first.battle_key,rolls)
    assert healed.hp[0] == 22 and healed.actor == 4 and healed.post_victory == 1
    assert not jnp.any(healed.last_xp)
    done,reward = attack(healed,jnp.int32(WAIT),healed.battle_key,rolls)
    assert not done.in_battle and done.post_victory == 0 and done.last_event == VICTORY
    assert reward == 1 and done.last_xp.sum() > 0 and not jnp.any(done.pending_healers)
    assert not jnp.any(done.weakened) and not jnp.any(done.paralyzed)


def test_cliric_red_post_victory_after_withdrawal_and_episode_deadline(current_game):
    from stoix.envs.number_grid import CONTINUE, WITHDRAW
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(33))).replace(actor=jnp.int32(0))
    start = enemy_roster_state(env,start,['squire','archer','archer','medium',None,None])
    start = start.replace(hp=start.hp.at[6].set(50),escaped=start.escaped.at[1:6].set(True),
        retreating=start.retreating.at[0].set(True))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    post,_ = attack(start,jnp.int32(CONTINUE),start.battle_key,rolls)
    assert post.in_battle and not post.lost and post.post_victory == 2 and post.actor == 9
    end,_ = attack(post,jnp.int32(CONTINUE),post.battle_key,rolls)
    assert not end.in_battle and end.last_event == WITHDRAW and not end.lost and end.last_damage == 25
    chex.assert_trees_all_equal(end.hp[:6],start.hp[:6])
    last = hero_roster_state(env,start,['archer','duke','possessed','medium','cultist',None])
    last = last.replace(actor=jnp.int32(0),retreating=jnp.zeros(12,bool),escaped=jnp.zeros(12,bool),
        hp=last.hp.at[6:].set(0).at[6].set(1),step_count=jnp.int32(env.max_steps-1))
    last = last.replace(potion_bonus=last.potion_bonus.at[0,2].set(100))
    out,ts = compiled_method(env,'step')(last,jnp.int32(SHOOT))
    assert out.done and ts.truncated() and out.in_battle and out.post_victory == 1
    assert out.actor == 3 and out.pending_healers[3]  # another player decision, never an automatic heal


def test_profit_mass_heal_includes_self_and_only_named_forms_cure(elemental_attack_game):
    """Discipl2.exe heal/cure canPerform and L_ALL target lists never exclude the caster."""
    env,_,start = elemental_attack_game
    assert MAP['enemy_rosters'][34][3] == 'cleric'
    start = start.replace(actor=jnp.int32(3),hp=start.hp.at[:5].set(jnp.array([50,100,0,5,20])),
        paralyzed=start.paralyzed.at[0].set(True),long_paralyzed=start.long_paralyzed.at[0].set(True),
        poison_mode=start.poison_mode.at[0].set(2),poison_damage=start.poison_damage.at[0].set(7),
        poison_source=start.poison_source.at[0].set(7),
        retreating=start.retreating.at[0].set(True).at[1].set(True),feared=start.feared.at[0].set(True),
        weakened=start.weakened.at[0].set(True),armor_shreds=start.armor_shreds.at[0].set(2),
        priority=jnp.zeros(12).at[6].set(10000.))
    names = ['cleric','abbess','prophetess','elf_oracle']
    ids = jnp.array([env.progression.ids[key] for key in names])
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(4,)+x.shape),start)
    states = states.replace(unit_ids=states.unit_ids.at[:,3].set(ids),unit_levels=states.unit_levels.at[:,3].set(env.progression.base_levels[ids]))
    rolls = jnp.full((4,env.random_size),.99)
    out,_ = scalar_battle(env)(states,jnp.full(4,SHOOT+3),states.battle_key,rolls)
    chex.assert_trees_all_equal(out.hp[:,0],jnp.array([70,90,120,110]))
    # The wounded caster (5 HP) heals itself by its own amount: 20/40/70/60.
    chex.assert_trees_all_equal(out.hp[:,3],jnp.array([25,45,75,65]))
    assert not jnp.any(out.hp[:,2]) and jnp.all(out.last_target == -1)
    chex.assert_trees_all_equal(out.paralyzed[:,0],jnp.array([1,0,0,1],bool))
    chex.assert_trees_all_equal(out.long_paralyzed[:,0],jnp.array([1,0,0,1],bool))
    chex.assert_trees_all_equal(out.retreating[:,0],jnp.ones(4,bool))
    assert jnp.all(out.retreating[:,1])  # Cure preserves a voluntary retreat.
    chex.assert_trees_all_equal(out.poison_source[:,0],jnp.array([7,-1,-1,7]))
    chex.assert_trees_all_equal(out.weakened[:,0],jnp.ones(4,bool))
    chex.assert_trees_all_equal(out.armor_shreds[:,0],jnp.full(4,2))


def test_sundancer_shared_ward_recast_owners_and_sticky_native_consumption():
    from stoix.envs.number_grid_wards import grant_wards,expire_wards,ward_tags
    from stoix.envs.number_grid_wards import WARD_MASK
    fire = jnp.uint32(4)  # Sundancer's GmodifL g000um0016 is a fire ward
    assert WARD_MASK == 270
    owners = jnp.zeros((12,4),jnp.uint32)
    sticky = jnp.zeros(12,jnp.uint32)
    native = jnp.zeros(12,jnp.uint32).at[1].set(fire).at[3].set(fire)
    used = jnp.zeros(12,jnp.uint32).at[1].set(fire)
    recipients = jnp.zeros(12,bool).at[1:4].set(True)
    grant,expire = jax.jit(grant_wards),jax.jit(expire_wards)
    owners,sticky,used = grant(owners,sticky,used,native,recipients,fire,jnp.int32(0))
    assert used[1] == 0 and sticky[1] == fire and jnp.all(ward_tags(owners)[1:4] == fire)
    owners,sticky,used = grant(owners,sticky,used,native,recipients,fire,jnp.int32(6))
    assert jnp.all(owners[1:4,0] == 65)
    owners,sticky,used = expire(owners,sticky,used,native,jnp.uint32(1))
    assert jnp.all(owners[1:4,0] == 64) and not jnp.any(used)
    # One blocked hit consumes the shared projection, independent of owner count.
    used = used.at[1:3].set(fire)
    owners,sticky,used = grant(owners,sticky,used,native,recipients,fire,jnp.int32(6))
    assert not jnp.any(used) and sticky[1] == fire
    owners,sticky,used = expire(owners,sticky,used,native,jnp.uint32(64))
    assert not jnp.any(owners) and not jnp.any(sticky)
    assert used[1] == fire and used[2] == 0 and used[3] == 0
    # Native protection first consumed while a temporary grant is active stays spent.
    owners,sticky,used = grant(owners,sticky,used,native,recipients,jnp.uint32(WARD_MASK),jnp.int32(11))
    used = used.at[3].set(WARD_MASK)
    owners,sticky,used = expire(owners,sticky,used,native,jnp.uint32(1 << 11))
    assert used[3] == fire and not jnp.any(owners)


def test_sundancer_wards_whole_party_with_fire_and_expires_on_wait_return(current_game):
    """GAttacks g000aa8134/g000aa8034: REACH=1 (all allies); GmodifL g000um0016: fire ward."""
    from stoix.envs.number_grid import CONTINUE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][35][4] == 'sundancer'
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','sundancer','cultist',None])
    state = enemy_roster_state(env,state,['archer','squire','squire','archer','archer','archer'])
    state = state.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    granted,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    # Every living ally, the caster included, receives the caster-owned fire ward.
    assert granted.last_damage == 0 and granted.last_target == -1
    assert jnp.all(granted.healer_wards[:5,0] == 8) and not jnp.any(granted.healer_wards[:5,1:])
    assert not jnp.any(granted.healer_wards[5])
    assert jnp.all((env.unit_traits(granted)[:5,3] & 4) == 4) and not jnp.any(env.unit_traits(granted)[:5,3] & 1)
    # A weapon hit passes through the fire ward. Force the enemy's lowest-HP target.
    target = granted.replace(actor=jnp.int32(6),hp=granted.hp.at[0].set(1),
        priority=jnp.zeros(12).at[1].set(10000.))
    hit,_ = attack(target,jnp.int32(CONTINUE),target.battle_key,rolls)
    assert hit.hp[0] == 0 and hit.last_ward == 0 and hit.wards_used[0] == 0
    # A fire area attack is blocked once on every warded ally.
    fire = enemy_roster_state(env,target,['cultist','squire','squire','archer','archer','archer'])
    blocked,_ = attack(fire,jnp.int32(CONTINUE),fire.battle_key,rolls)
    chex.assert_trees_all_equal(blocked.hp[:5],fire.hp[:5])
    assert blocked.last_ward == 0b11111 and jnp.all(blocked.wards_used[:5] & 4)
    assert blocked.healer_wards[0,0] == 8
    # Returning after WAIT expires the caster's grants even after first activation.
    waiting = granted.replace(actor=jnp.int32(4),turn_phase=jnp.full(12,2,jnp.int32).at[3].set(1),
        priority=granted.priority.at[3].set(-1.),
        activation_done=jnp.ones(12,bool),paralyzed=granted.paralyzed.at[3].set(True))
    expired,_ = attack(waiting,jnp.int32(DEFEND),waiting.battle_key,rolls)
    assert expired.actor == 3 and not jnp.any(expired.healer_wards)
    assert not jnp.any(env.unit_traits(expired)[:5,3] & 4)


def test_sundancer_expiry_in_saved_forms_and_on_caster_lethal_periodic_tick(current_game):
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['possessed','duke','possessed','sundancer','cultist',None])
    # An original ward snapshot must lose caster ownership while its unit is an imp.
    saved = start.replace(imp=start.imp.at[0].set(True),
        saved_healer_wards=start.saved_healer_wards.at[0,0].set(8),
        saved_ward_native_used=start.saved_ward_native_used.at[0].set(0))
    expired,used = compiled_method(env,'_expire_healer_wards')(saved,saved.wards_used,jnp.uint32(8))
    restored = compiled_method(env,'_start_activation')(expired.replace(actor=jnp.int32(0),activation_done=jnp.zeros(12,bool)),jnp.float32(0))
    assert not restored.imp[0] and not jnp.any(restored.healer_wards) and not jnp.any(restored.saved_healer_wards)
    # Caster grants expire on its activation, even if poison kills it before acting.
    poisoned = start.replace(actor=jnp.int32(4),hp=start.hp.at[3].set(1),
        healer_wards=start.healer_wards.at[0,0].set(8),priority=jnp.zeros(12).at[3].set(10000.).at[1].set(100.),
        activation_done=jnp.zeros(12,bool),poison_mode=start.poison_mode.at[3].set(2),
        poison_damage=start.poison_damage.at[3].set(20))
    end,_ = attack(poisoned,jnp.int32(DEFEND),poisoned.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    assert end.hp[3] == 0 and end.actor == 1 and not jnp.any(end.healer_wards)


def test_sylfid_mass_heal_air_ward_and_minimum_hp_representative(current_game):
    """GAttacks: Sylfid wards and heal both have REACH=1, so they cover the caster too."""
    from stoix.envs.number_grid_combat import UNITS
    from stoix.envs.number_grid import CONTINUE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    p = UNITS['sylfid']
    assert MAP['enemy_rosters'][36][5] == 'sylfid'
    assert tuple(p[k] for k in ('max_hp','damage','initiative','exp_kill','exp_required')) == (95,55,10,230,2470)
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','sylfid','cultist',None])
    state = enemy_roster_state(env,state,['apprentice','squire','squire','archer','archer','archer'])
    state = state.replace(actor=jnp.int32(3),hp=state.hp.at[0].set(50).at[3].set(30),
                          priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    healed,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    # Both wounded units heal by 55, the caster included; the sum is reported.
    assert healed.hp[0] == 105 and healed.hp[3] == 85 and healed.last_damage == 110
    assert jnp.all(healed.healer_wards[:5,1] == 8) and not jnp.any(healed.healer_wards[5])
    assert env.unit_traits(healed)[0,3] == (1 << 8)
    attacked,_ = attack(healed.replace(actor=jnp.int32(6),priority=jnp.zeros(12).at[1].set(10000.)),jnp.int32(CONTINUE),healed.battle_key,rolls)
    assert attacked.hp[0] == 105 and attacked.last_ward & 1 and attacked.wards_used[0] == (1 << 8)
    # The apprentice's air area attack is blocked on every warded ally, the caster too.
    chex.assert_trees_all_equal(attacked.hp[:5],healed.hp[:5])
    assert attacked.wards_used[3] == (1 << 8)


def test_travnitsa_transformed_original_buff_expires_with_imp_turn(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import DAMAGE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['duke','lord','possessed','travnitsa',None,'cultist'])
    start = enemy_roster_state(env,start,['witch','squire','squire','archer','archer','archer'])
    start = start.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    buffed,_ = attack(start,jnp.int32(SHOOT+1),start.battle_key,rolls)
    imp,_ = attack(buffed.replace(actor=jnp.int32(6)),jnp.int32(CONTINUE),buffed.battle_key,rolls)
    assert imp.imp[1] and imp.saved_powerup[1] and imp.saved_primary_override[1] == 212
    assert not imp.powerup[1] and compiled_method(env,'unit_stats')(imp)[1,DAMAGE] == 30
    spent,_ = attack(imp.replace(actor=jnp.int32(1)),jnp.int32(DEFEND),imp.battle_key,rolls)
    assert not spent.saved_powerup[1] and spent.saved_primary_override[1] == 170
    restored = compiled_method(env,'_start_activation')(spent.replace(actor=jnp.int32(1),activation_done=jnp.zeros(12,bool)),jnp.float32(0))
    assert not restored.imp[1] and compiled_method(env,'unit_stats')(restored)[1,DAMAGE] == 170


def test_travnitsa_buff_lasts_through_two_strikes_but_expires_on_paralysis(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.envs.number_grid_combat import DAMAGE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['blade_master','duke','possessed','travnitsa','cultist',None])
    start = enemy_roster_state(env,start,['tiamat','squire','squire',None,'archer','archer'])
    start = start.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[1].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    original = compiled_method(env,'unit_stats')(start)[0,DAMAGE]
    buffed,_ = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    first,_ = attack(buffed.replace(actor=jnp.int32(0)),jnp.int32(SHOOT),buffed.battle_key,rolls)
    assert first.second_strike and first.powerup[0] and first.actor == 0
    second,_ = attack(first,jnp.int32(SHOOT),first.battle_key,rolls)
    assert not second.second_strike and not second.powerup[0] and compiled_method(env,'unit_stats')(second)[0,DAMAGE] == original
    blocked = buffed.replace(actor=jnp.int32(0),paralyzed=buffed.paralyzed.at[0].set(True))
    skipped,_ = attack(blocked,jnp.int32(CONTINUE),blocked.battle_key,rolls)
    assert not skipped.powerup[0] and compiled_method(env,'unit_stats')(skipped)[0,DAMAGE] == original


def test_travnitsa_fenrir_uses_reference_stored_original_damage(current_game):
    from stoix.envs.number_grid import FENRIR
    from stoix.envs.number_grid_combat import DAMAGE
    env,initial,_,attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['wolf_lord','duke','possessed','travnitsa','cultist',None])
    state = state.replace(actor=jnp.int32(0),priority=jnp.zeros(12).at[1].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    wolf,_ = attack(state,jnp.int32(FENRIR),state.battle_key,rolls)
    assert wolf.fenrir[0] and compiled_method(env,'unit_stats')(wolf)[0,DAMAGE] == 90
    # battle_env.py6991 does not replace original_damage=40 when entering Fenrir.
    buffed,_ = attack(wolf.replace(actor=jnp.int32(3)),jnp.int32(SHOOT),wolf.battle_key,rolls)
    assert compiled_method(env,'unit_stats')(buffed)[0,DAMAGE] == 50
    spent,_ = attack(buffed.replace(actor=jnp.int32(0)),jnp.int32(DEFEND),buffed.battle_key,rolls)
    assert compiled_method(env,'unit_stats')(spent)[0,DAMAGE] == 40 and not spent.powerup[0]


def test_deva_roshi_heals_and_wards_one_target_with_four_elements_without_cure(current_game):
    """GAttacks g000aa8133/g000aa8033: REACH=2 (one ally, the caster allowed).

    GmodifL wards g000um0001/0016/0010/0005: air, fire, earth and water.
    """
    from stoix.envs.number_grid_combat import UNITS
    from stoix.envs.number_grid_wards import WARD_MASK
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][38][4] == 'deva_roshi'
    assert UNITS['deva_roshi']['max_hp'] == 85 and UNITS['deva_roshi']['damage'] == 50
    assert not env.progression.mass_healers[env.progression.ids['deva_roshi']]
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['possessed','duke','possessed','deva_roshi','cultist',None])
    start = start.replace(actor=jnp.int32(3),hp=start.hp.at[0].set(1).at[2].set(0).at[3].set(5),
        paralyzed=start.paralyzed.at[0].set(True),priority=jnp.zeros(12).at[6].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    own,_ = attack(start,jnp.int32(SHOOT+3),start.battle_key,rolls)
    assert own.hp[3] == 55 and own.hp[0] == 1 and own.hp[2] == 0 and own.last_target == 3
    assert jnp.all(own.healer_wards[3] == 8) and (env.unit_traits(own)[3,3] & WARD_MASK) == WARD_MASK
    assert not jnp.any(own.healer_wards[:3]) and not jnp.any(own.healer_wards[4:])
    ally,_ = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    assert ally.hp[0] == 51 and ally.hp[3] == 5 and ally.paralyzed[0] and ally.last_target == 0
    assert jnp.all(ally.healer_wards[0] == 8) and not jnp.any(ally.healer_wards[1:])
    assert (env.unit_traits(ally)[0,3] & WARD_MASK) == WARD_MASK


def test_enemy_single_ward_healer_still_wards_without_wounded_allies(current_game):
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,_ = current_game
    red = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    red = enemy_roster_state(env,red,['squire','archer','squire','deva_roshi','archer',None])
    red = red.replace(actor=jnp.int32(9),hp=red.hp.at[7].set(30))
    choose = compiled_method(env,'_enemy_action')
    # A wounded ally is healed first; otherwise the lowest-HP living ally gets wards.
    assert choose(red,jnp.zeros(6)) == SHOOT+1
    full = enemy_roster_state(env,red,['squire','archer','squire','deva_roshi','archer',None]).replace(actor=jnp.int32(9))
    assert choose(full,jnp.zeros(6)) == SHOOT+1  # archer 45 HP < Deva roshi 85 < squire 100


@pytest.mark.core
def test_patriach_revives_once_half_hp_cleanses_and_cannot_act_this_round(current_game):
    env,initial,advance,attack = current_game
    assert MAP['enemy_rosters'][39][4] == 'patriarch'
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','patriarch','cultist',None])
    state = state.replace(actor=jnp.int32(3),hp=state.hp.at[4].set(0),
        paralyzed=state.paralyzed.at[4].set(True),long_paralyzed=state.long_paralyzed.at[4].set(True),
        feared=state.feared.at[4].set(True),retreating=state.retreating.at[4].set(True),
        poison_mode=state.poison_mode.at[4].set(2),poison_damage=state.poison_damage.at[4].set(20),
        poison_source=state.poison_source.at[4].set(6),battle_xp=jnp.array([0,100],jnp.int32),
        priority=jnp.zeros(12).at[1].set(10000.))
    assert compiled_method(env,'action_mask')(state)[SHOOT+4]
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    risen,_ = attack(state,jnp.int32(SHOOT+4),state.battle_key,rolls)
    assert risen.hp[4] == 22  # Python round(45*.5), ties to even.
    assert risen.battle_revived[4] and risen.battle_earned_xp[4] == 0 and risen.turn_phase[4] == 2
    assert not risen.paralyzed[4] and not risen.long_paralyzed[4] and not risen.feared[4] and not risen.retreating[4]
    assert risen.poison_mode[4] == 0 and risen.poison_source[4] == -1
    fallen = risen.replace(actor=jnp.int32(3),hp=risen.hp.at[4].set(0))
    assert not compiled_method(env,'action_mask')(fallen)[SHOOT+4]
    invalid,_ = advance(fallen,jnp.int32(SHOOT+4))
    assert invalid.hp[4] == 0
    chex.assert_trees_all_equal(invalid.battle_key,fallen.battle_key)
    # Healing a living unit does not grant resurrection's cleansing.
    wounded = state.replace(hp=state.hp.at[4].set(1))
    healed,_ = attack(wounded,jnp.int32(SHOOT+4),wounded.battle_key,rolls)
    assert healed.hp[4] == 45 and healed.paralyzed[4] and healed.poison_mode[4] == 2


def test_patriach_large_footprint_and_ai_dead_first_uniform_ties(current_game):
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = enemy_roster_state(env,state,['titan','squire','squire','archer','patriarch','archer'])
    state = state.replace(actor=jnp.int32(10),hp=state.hp.at[6].set(0))
    # A large corpse cannot overlap a living unit in its rear paired cell.
    assert not compiled_method(env,'_patriarch_targets')(state)[6]
    rear_dead = state.replace(hp=state.hp.at[9].set(0))
    assert compiled_method(env,'_patriarch_targets')(rear_dead)[6]
    big_alive = state.replace(hp=state.hp.at[6].set(250).at[9].set(0))
    assert not compiled_method(env,'_patriarch_targets')(big_alive)[9]
    two_dead = rear_dead.replace(hp=rear_dead.hp.at[7].set(0),battle_revived=rear_dead.battle_revived.at[9].set(True))
    ai = compiled_method(env,'_enemy_action')
    assert ai(two_dead,jnp.array([.1,.9,0,0,0,0])) == SHOOT
    assert ai(two_dead,jnp.array([.9,.1,0,0,0,0])) == SHOOT+1
    used = two_dead.replace(battle_revived=jnp.ones(12,bool),hp=two_dead.hp.at[8].set(1))
    assert ai(used,jnp.zeros(6)) == SHOOT+2


def test_patriach_post_victory_revival_receives_no_past_xp(current_game):
    from stoix.envs.number_grid import WAIT
    env,initial,_,attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','patriarch','cultist',None])
    state = state.replace(actor=jnp.int32(3),post_victory=jnp.int32(1),
        pending_healers=jnp.zeros(12,bool).at[3].set(True),
        hp=state.hp.at[4:].set(0),battle_xp=jnp.array([0,100],jnp.int32),
        battle_earned_xp=state.battle_earned_xp.at[:4].set(25))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    risen,_ = attack(state,jnp.int32(SHOOT+4),state.battle_key,rolls)
    assert not risen.in_battle and risen.hp[4] == 22 and risen.last_xp[4] == 1  # final minimum, not old kills
    chex.assert_trees_all_equal(risen.last_xp[:4],jnp.full(4,25,jnp.int32))
    skipped,_ = attack(state,jnp.int32(WAIT),state.battle_key,rolls)
    assert skipped.hp[4] == 0 and jnp.sum(skipped.last_xp) == 100


def test_novice_half_extra_primary_damage_without_stacking(current_game):
    from stoix.envs.number_grid_combat import DAMAGE
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][40][4] == 'novice'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['knight','duke','possessed','novice','cultist',None])
    start = start.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[1].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    buffed,_ = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    assert compiled_method(env,'unit_stats')(buffed)[0,DAMAGE] == 75 and buffed.powerup[0]
    recast,_ = attack(buffed.replace(actor=jnp.int32(3)),jnp.int32(SHOOT),buffed.battle_key,rolls)
    assert compiled_method(env,'unit_stats')(recast)[0,DAMAGE] == 75


@pytest.mark.core
def test_alchemist_extra_turn_preserves_wait_and_round_effects(current_game):
    from stoix.envs.number_grid import WAIT
    env,initial,advance,attack = current_game
    assert MAP['enemy_rosters'][20][4] == 'alchemist'
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['knight','duke','possessed','alchemist','cultist',None])
    start = start.replace(actor=jnp.int32(3),turn_phase=jnp.full(12,2,jnp.int32).at[6].set(0).at[3].set(0),
        waited=start.waited.at[0].set(True),activation_done=jnp.ones(12,bool),
        poison_mode=start.poison_mode.at[0].set(2),poison_damage=start.poison_damage.at[0].set(20),
        priority=jnp.zeros(12).at[6].set(5.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    extra,_ = attack(start,jnp.int32(SHOOT),start.battle_key,rolls)
    assert extra.actor == 0 and extra.turn_phase[0] == 0 and extra.priority[0] == 50
    assert extra.hp[0] == 150 and extra.poison_mode[0] == 2 and extra.bonus_turns[0] == 1
    assert extra.waited[0] and not compiled_method(env,'action_mask')(extra)[WAIT]
    invalid,_ = advance(extra,jnp.int32(WAIT))
    chex.assert_trees_all_equal(invalid.battle_key,extra.battle_key)
    # A granted turn still permits two consecutive strikes from a dual attacker.
    dual = hero_roster_state(env,start,['blade_master','duke','possessed','alchemist','cultist',None])
    # Spending WAIT earlier does not reduce the attack count of a newly granted turn.
    dual = dual.replace(poison_mode=jnp.zeros(12,jnp.int32))
    granted,_ = attack(dual,jnp.int32(SHOOT),dual.battle_key,rolls)
    first,_ = attack(granted,jnp.int32(SHOOT),granted.battle_key,rolls)
    assert first.second_strike and first.actor == 0
    assert granted.waited[0] and not compiled_method(env,'action_mask')(granted)[WAIT]


def test_alchemist_rejects_self_other_alchemist_retreat_and_ai_selects_spent_damage(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    start = hero_roster_state(env,start,['knight','duke','possessed','alchemist','alchemist',None])
    start = start.replace(actor=jnp.int32(3),retreating=start.retreating.at[2].set(True))
    mask = compiled_method(env,'action_mask')(start)
    assert mask[SHOOT] and mask[SHOOT+1] and not jnp.any(mask[SHOOT+2:SHOOT+6])
    red = enemy_roster_state(env,start,['squire','lord','squire','alchemist',None,'archer'])
    red = red.replace(actor=jnp.int32(9),turn_phase=red.turn_phase.at[7].set(2),priority=jnp.full(12,1000.))
    ai = compiled_method(env,'_enemy_action')
    assert ai(red,jnp.zeros(6)) == SHOOT+1
    rested = red.replace(turn_phase=jnp.zeros(12,jnp.int32))
    assert ai(rested,jnp.zeros(6)) == DEFEND
    result,_ = attack(red,jnp.int32(CONTINUE),red.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    assert result.bonus_turns[7] == 1


def test_hermit_secondary_miss_and_primary_water_protection(elemental_attack_game):
    env,_,state = elemental_attack_game
    state = hero_roster_state(env,state,['possessed','duke','possessed','hermit','cultist',None]).replace(actor=jnp.int32(3))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    attack = compiled_method(env,'_battle_step')
    hit,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    assert hit.hp[6] == 300 and hit.hp[7] == 300 and jnp.all(hit.hp[8:] == 245)
    assert hit.slow_original[6] == -1 and hit.slow_original[7] == -1 and jnp.all(hit.slow_original[8:] >= 0)
    miss,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls.at[37:49].set(.99))
    chex.assert_trees_all_equal(miss.hp,hit.hp)
    assert jnp.all(miss.slow_original == -1)


@pytest.mark.core
def test_vampire_nosferatu_area_actual_damage_leech_self_only_and_hero_growth(current_game):
    from stoix.envs.number_grid_combat import UNITS
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][33][0] == 'nosferatu' and UNITS['nosferatu']['hero']
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['nosferatu','duke','possessed','cultist','cultist',None])
    state = enemy_roster_state(env,state,['squire','squire','titan','archer','archer',None])
    state = state.replace(actor=jnp.int32(0),hp=jnp.array([1,100,100,10,10,0,5,9,250,30,30,0],jnp.int32),
        priority=jnp.zeros(12).at[1].set(10000.))
    result,_ = attack(state,jnp.int32(SHOOT),state.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    assert result.hp[0] == 23 and result.last_damage == 44 and result.last_target == -1
    chex.assert_trees_all_equal(result.hp[1:6],state.hp[1:6])
    unit_id = env.progression.ids['nosferatu']
    assert env.progression.aoe_accuracy_falloff[unit_id]
    assert env.progression.required_xp(jnp.int32(unit_id),jnp.int32(2)) == 550
    assert env.progression.experience(jnp.int32(unit_id),jnp.int32(2),jnp.int32(0))[1] == 44


def test_highvampire_leech_heals_self_then_splits_leftover_with_caps(current_game):
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][34][4] == 'highvampire'
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['highvampire','duke','possessed','cultist','cultist',None])
    state = enemy_roster_state(env,state,['squire','squire','titan','archer','archer',None])
    state = state.replace(actor=jnp.int32(0),hp=jnp.array([209,100,100,10,10,0,5,9,250,30,30,0],jnp.int32),
        priority=jnp.zeros(12).at[1].set(10000.))
    result,_ = attack(state,jnp.int32(SHOOT),state.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    assert result.last_damage == 134
    chex.assert_trees_all_equal(result.hp[:6],jnp.array([210,117,117,26,26,0]))


def test_drulliaan_life_area_and_independent_uncached_water_effect(elemental_attack_game):
    env,_,state = elemental_attack_game
    assert MAP['enemy_rosters'][36][0] == 'drulliaan'
    state = hero_roster_state(env,state,['possessed','duke','possessed','drulliaan','cultist',None]).replace(actor=jnp.int32(3))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    result,_ = compiled_method(env,'_battle_step')(state,jnp.int32(SHOOT),state.battle_key,rolls)
    chex.assert_trees_all_equal(result.hp[6:],jnp.full(6,125,jnp.int32))
    chex.assert_trees_all_equal(result.water_mode[6:],jnp.array([0,0,2,2,2,2]))
    assert jnp.all(result.water_source == -1) and jnp.all(result.water_damage[8:] == 30)
    # Miss the separate status roll while retaining the primary Life damage.
    miss_rolls = rolls.at[37:49].set(.99)
    miss,_ = compiled_method(env,'_battle_step')(state,jnp.int32(SHOOT),state.battle_key,miss_rolls)
    chex.assert_trees_all_equal(miss.hp[6:],result.hp[6:])
    assert not jnp.any(miss.water_mode)


def test_elfarcher_two_ranged_strikes_can_retarget_rear(current_game):
    from stoix.envs.number_grid import WAIT
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][37][0] == 'elf_bandit'
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','elf_bandit','cultist',None])
    state = enemy_roster_state(env,state,['squire','squire','squire','archer','archer','archer'])
    state = state.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[1].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    first,_ = attack(state,jnp.int32(SHOOT+5),state.battle_key,rolls)
    assert first.hp[11] == state.hp[11]-30 and first.second_strike and first.actor == 3
    mask = compiled_method(env,'action_mask')(first)
    assert mask[SHOOT+4] and not mask[WAIT] and not mask[DEFEND]
    second,_ = attack(first,jnp.int32(SHOOT+4),first.battle_key,rolls)
    assert second.hp[10] == state.hp[10]-30 and not second.second_strike


def test_shamanka_earth_damage_then_separate_mind_fear_not_primary_fear(elemental_attack_game):
    env,initial,_ = elemental_attack_game
    assert MAP['enemy_rosters'][38][0] == 'shamanka'
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(22)))
    state = hero_roster_state(env,state,['possessed','duke','possessed','shamanka','cultist',None])
    state = state.replace(actor=jnp.int32(3),priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99)
    attack = compiled_method(env,'_battle_step')
    result,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    chex.assert_trees_all_equal(result.hp[6:],jnp.full(6,970,jnp.int32))
    chex.assert_trees_all_equal(result.paralyzed[6:],jnp.array([False,False,True,True,True,True]))
    assert not jnp.any(result.retreating[6:])  # This fixture protects RED's interior.
    missed,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls.at[37:49].set(.99))
    chex.assert_trees_all_equal(missed.hp[6:],result.hp[6:])
    assert not jnp.any(missed.paralyzed)


def test_shamanka_enemy_uses_regular_area_ai_and_fear_causes_escape(current_game):
    from stoix.envs.number_grid import CONTINUE
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    initial = hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist',None])
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = enemy_roster_state(env,state,['shamanka','squire','squire','archer','archer','archer'])
    state = state.replace(actor=jnp.int32(6),hp=state.hp.at[:5].set(jnp.array([100,150,120,40,45])),
        priority=jnp.zeros(12).at[7].set(10000.))
    assert compiled_method(env,'_enemy_action')(state,jnp.zeros(6)) == SHOOT+3
    hit,_ = attack(state,jnp.int32(CONTINUE),state.battle_key,jnp.zeros(env.random_size).at[36].set(.99))
    assert jnp.all(hit.feared[:5]) and jnp.all(hit.retreating[:5]) and not jnp.any(hit.escaped[:5])


def test_aleman_melee_shatter_ignores_secondary_power_roll_requires_positive_hit(current_game):
    from stoix.envs.number_grid_combat import ARMOR, UNITS
    from stoix.tests.number_grid_fixtures import enemy_roster_state
    env,initial,_,attack = current_game
    assert MAP['enemy_rosters'][39][0] == 'aleman'
    p = UNITS['aleman']
    assert tuple(p[k] for k in ('max_hp','damage','accuracy','armor','initiative')) == (800,300,80,20,60)
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['possessed','aleman','possessed','cultist','cultist',None])
    state = enemy_roster_state(env,state,['drulliaan','squire','squire','archer','archer','archer'])
    state = state.replace(actor=jnp.int32(1),priority=jnp.zeros(12).at[0].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36:49].set(.99)
    result,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    assert result.hp[6] == 1850 and result.armor_shreds[6] == 1 and compiled_method(env,'unit_stats')(result)[6,ARMOR] == 35
    missed,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls.at[12:24].set(.99))
    assert missed.hp[6] == 2000 and missed.armor_shreds[6] == 0
