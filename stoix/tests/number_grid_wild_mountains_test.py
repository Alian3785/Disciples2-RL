"""S143 «Дикие горы»: source, start, weakened guardians and agent garrison battles; CUDA, by request only."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, NumberGrid, action_names, load_map
from stoix.envs.number_grid_combat import ACCURACY, ARMOR, DAMAGE, HP, INITIATIVE
from stoix.envs.number_grid_casting import DAMAGE as DAMAGE_SPELL, DEBUFF, SUMMON as SUMMON_SPELL
from stoix.envs.number_grid_spells import knows
from stoix.envs.number_grid_thief import THIEF_FAILED, THIEF_POISONED
from stoix.envs import number_grid_reference_rewards as reference_rewards
from stoix.tests.number_grid_fixtures import compiled_method

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT/'docs/maps/wild_mountains/source.json'
WEAKENED = dict(max_hp=190, armor=0, damage=90, accuracy=78, initiative=41)


@pytest.fixture(scope='module', autouse=True)
def cuda_only():
    assert jax.default_backend() == 'gpu', 'Run only on request through scripts/run_gpu.sh'


@pytest.fixture(scope='module')
def mountains():
    env = NumberGrid(map_config=load_map('wild_mountains'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    return env, state


@pytest.fixture(scope='module')
def mountains_rehire():
    """The separate no-capital-attacks variant still allows replacing a destroyed party."""
    env = NumberGrid(map_config=load_map('wild_mountains_no_capital_attacks'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    return env, state


def enemy_index(env, source_id):
    return env.map_config['reference_enemy_ids'].index(source_id)


def test_source_rebuilds_map_without_thieves():
    from scripts.build_wild_mountains import build, truncated_percent
    source = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    config = load_map('wild_mountains')
    assert build(source) == config
    ids = config['reference_enemy_ids']
    assert len(ids) == 190 and 'S143KC007b' not in ids and 'S143KC007c' not in ids
    assert 'S143KC0000' not in ids and 'S143FT0000' not in ids
    # Shornhelm; the .sg spells several city names with Latin look-alike letters.
    assert len(config['cities']) == 8 and [c['source_id'] for c in config['cities'] if c['objective']] == ['S143FT0006']
    assert [c['position'] for c in config['cities'] if c['objective']] == [[68, 66]]
    assert len(config['chests']) == 71 and sum('guards' in c for c in config['chests']) == 55
    assert len(config['ruins']) == 12 and len(config['mines']) == 17
    assert len(config['merchants']) == 2 and len(config['trainers']) == 1
    assert [len(s['spells']) for s in config['spell_shops']] == [13, 6]  # Daylight omitted
    assert [o['unit'] for o in config['mercenaries'][0]['roster']] == ['g000uu5023', 'g000uu8051', 'g000uu5022']
    capital, = config['enemy_capitals']
    assert capital['name'] == 'Сэнтинел' and [ids[g] for g in capital['guards']] == ['S143FT0001', 'S143KC0001']
    assert 'scenario_events' not in config and config['scheduled_enemies'] == []
    assert config['capital_defender'] == dict(unit='g000uu3003', level=1, combat_stats=WEAKENED)
    assert config['defeat'] == 'party_or_capital' and config['leader_hire'] == dict(unit='duke')
    # Seven -20% modifiers: 900 -> 190, the HP stored in the .sg.
    hp = 900
    for _ in range(7):
        hp = truncated_percent(hp, -20)
    assert hp == 190
    assert (len(source['events']), len(source['stack_templates']), len(source['locations'])) == (48, 39, 23)


def test_start_party_potions_treasury_and_guardian(mountains):
    env, state = mountains
    rows = env.progression.rows
    assert state.position.tolist() == [7, 20]
    assert [rows[int(i)]['name'] for i in state.unit_ids[:6]] == [
        'Одержимый', 'Мидредэль', 'Пусто', 'Пусто', 'Сектант', 'Пусто']
    assert state.unit_levels[:6].tolist() == [1, 1, 0, 0, 1, 0]
    assert state.hp[:6].tolist() == [120, 150, 0, 0, 45, 0]
    assert int(state.potions[env.potion_rules.keys.index('restoration')]) == 3
    assert int(state.potions[env.potion_rules.keys.index('life')]) == 5
    assert int(jnp.sum(state.potions)) == 8
    assert int(state.gold) == 250 and state.mana.tolist() == [75, 0, 0, 0, 0]
    assert env.recruitment.composition(state)[:3].tolist() == [3, 2, 1]
    g = env.garrisons
    capital = g.capital_index
    # Ashkael alone in the capital garrison, in the rear middle cell (user rule).
    assert state.garrison_ids[capital].tolist() == [0, 0, 0, 0, g.guardian, 0]
    assert state.garrison_hp[capital].tolist() == [0, 0, 0, 0, 190, 0] and g.guardian_max_hp == 190
    stats = env.progression.stats(jnp.int32(g.guardian), jnp.int32(1))
    assert [float(stats[i]) for i in (HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE)] == [190, 90, 78, 0, 41]
    assert int(jnp.sum(state.garrison_ids[:capital] != 0)) == 0 and g.cities == 8
    assert env.territory.objective_cities and not env.script_events.enabled and env.campaign_events.enabled
    assert state.learned_spells.shape == (2,)  # 17 Legions spells + 17 tower spells


def test_tower_spell_past_index_31_is_learned_in_the_second_word(mountains):
    """SiteRules and SpellCastRules share the book order, also beyond the first 32 spells."""
    env, state = mountains
    shop, casting = env.site_rules, env.casting
    step, mask = compiled_method(env, 'step'), compiled_method(env, 'action_mask')
    wide = [k for k, o in enumerate(shop.spell_offers) if casting_row(casting, o['spell']) >= 32]
    assert wide
    for k in wide:
        offer = shop.spell_offers[k]
        index = casting_row(casting, offer['spell'])
        assert index < casting.book_count
        at_shop = state.replace(position=shop.approach_tiles[offer['site'], 0], gold=jnp.int32(offer['price']))
        assert not bool(knows(at_shop.learned_spells, casting.book_bits[index]))
        buy = jnp.int32(shop.spell_start+k)
        assert bool(mask(at_shop)[buy])
        bought, _ = step(at_shop, buy)
        assert (bought.learned_spells ^ at_shop.learned_spells).tolist() == [0, 1 << (index-32)]
        assert bool(knows(bought.learned_spells, casting.book_bits[index]))
        assert not bool(mask(bought)[buy])


def test_undead_capital_is_passive_and_weakened(mountains):
    env, state = mountains
    guardian = enemy_index(env, 'S143FT0001')
    ids = env.progression.enemy_ids[guardian]
    names = [env.progression.rows[int(i)]['name'] for i in ids]
    ashgan = names.index('Ашган')
    stats = env.progression.stats(ids[ashgan], jnp.int32(1))
    assert [float(stats[i]) for i in (HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE)] == [190, 90, 78, 0, 41]
    assert bool(env.territory.capital_guards[guardian]) and bool(env.territory.capital_guards[enemy_index(env, 'S143KC0001')])
    rested, _ = compiled_method(env, 'step')(state, jnp.int32(REST))
    native = slice(0, 164)  # event armies may appear; the native groups stay
    assert jnp.array_equal(rested.enemy_positions[native], state.enemy_positions[native]) and not bool(rested.in_battle)


def fight(env, state, limit=400):
    step, mask = compiled_method(env, 'step'), compiled_method(env, 'action_mask')
    total = 0.
    for _ in range(limit):
        if not bool(state.in_battle) or bool(state.done):
            break
        legal = mask(state)
        choices = [a for a in range(SHOOT, DEFEND+1) if bool(legal[a])]
        state, ts = step(state, jnp.int32(choices[0] if choices else int(jnp.argmax(legal))))
        total += float(ts.reward)
    return state, total


def capital_cell(env):
    return env.garrisons.positions[env.garrisons.capital_index]


def attacked(env, state, source_id):
    """REST with the party away and one army moved onto the capital entrance."""
    enemy = enemy_index(env, source_id)
    before = state.replace(position=jnp.array([12, 20], jnp.int32),
                           enemy_positions=state.enemy_positions.at[enemy].set(capital_cell(env)))
    after, _ = compiled_method(env, 'step')(before, jnp.int32(REST))
    return enemy, before, after


def test_garrison_battle_is_played_by_the_agent(mountains):
    env, state = mountains
    capital = env.garrisons.capital_index
    enemy, before, battle = attacked(env, state, 'S143KC0033')  # goblins
    assert bool(battle.in_battle & battle.garrison_battle) and int(battle.enemy) == enemy
    assert int(battle.garrison_site) == capital
    assert battle.unit_ids[:6].tolist() == state.garrison_ids[capital].tolist()
    # Faster goblins may strike before the agent's first decision.
    assert 0 < int(battle.hp[4]) <= 190 and int(battle.battle_hero_fort_armor) == 50
    assert not bool(env._can_retreat(battle, jnp.int32(0)))
    assert battle.position.tolist() == [12, 20]
    end, _ = fight(env, battle)
    assert not bool(end.in_battle | end.garrison_battle | end.lost)
    assert not bool(end.alive[enemy]) and 0 < int(end.garrison_hp[capital, 4]) <= 190
    for name in ('unit_ids', 'unit_levels', 'unit_xp', 'hp'):
        assert jnp.array_equal(getattr(end, name)[:6], getattr(before, name)[:6]), name
    for name in ('equipped', 'battle_items', 'item_inventory', 'potions', 'position', 'city_owned'):
        assert jnp.array_equal(getattr(end, name), getattr(before, name)), name


def test_fallen_guardian_loses_the_capital(mountains):
    env, state = mountains
    # REST regenerates Ashkael fully (100%); the Red and White Dragons still kill him.
    enemy, before, battle = attacked(env, state, 'S143KC0051')
    assert int(battle.hp[4]) <= 190
    end, _ = fight(env, battle)
    assert bool(end.lost & end.done) and int(end.garrison_hp[env.garrisons.capital_index, 4]) == 0
    assert not bool(end.garrison_battle) and bool(end.alive[enemy])
    assert jnp.array_equal(end.unit_ids[:6], before.unit_ids[:6])


def destroyed(env, state, position, rewards=False):
    """The Black and Green Dragons destroy the starting party in a battle at position."""
    dragons = enemy_index(env, 'S143KC0052')
    start = state.replace(position=jnp.array(position, jnp.int32), enemy=jnp.int32(dragons))
    end, total = fight(env, jax.jit(lambda s: env._begin_battle(s))(start))
    return (dragons, end, total) if rewards else (dragons, end)


def test_party_loss_has_no_penalty_of_its_own(mountains_rehire):
    """A destroyed party keeps its episode record and multiplier, without a separate loss penalty."""
    env, state = mountains_rehire
    assert env.map_config['leader_hire'] == dict(unit='duke') and int(state.party_losses) == 0
    up = state.replace(unit_levels=state.unit_levels.at[:6].set(jnp.where(state.unit_ids[:6] != 0, 3, 0)))
    up = env.defense_rewards.reset(up)
    _, end, total = destroyed(env, up, [30, 30], rewards=True)
    assert int(end.party_losses) == 1 and int(end.party_level_record) == int(up.party_level_record)
    assert float(end.party_reward_multiplier) == 2.
    _, reward = env.defense_rewards.apply(env, up, end)
    assert float(reward) == 0.


def test_defense_rewards(mountains):
    """Terminal 0/-10, unpaid capital attackers, direct party level records."""
    env, state = mountains
    rules = env.defense_rewards
    config = env.map_config
    assert (config['timeout_reward'], config['defeat_reward']) == (0., -10.)
    assert (rules.attacker_xp, rules.party_level_record, rules.mana_guard) == (0., 2., .5)
    levels = state.unit_levels[:6]
    living = (state.unit_ids[:6] != 0) & (state.hp[:6] > 0)
    assert int(state.party_level_record) == int(jnp.sum(jnp.where(living, levels, 0)))
    # A level-up on the map directly pays +2; losing and regaining it cannot pay again.
    hero = int(jnp.argmax(env.progression.heroes[state.unit_ids[:6]]))
    up = state.replace(unit_levels=state.unit_levels.at[hero].add(1))
    paid_up, reward = rules.apply(env, state, up)
    assert float(reward) == 2.
    down, reward = rules.apply(env, paid_up, paid_up.replace(hp=paid_up.hp.at[hero].set(0)))
    assert float(reward) == 0. and int(down.party_level_record) == int(paid_up.party_level_record)
    restored, reward = rules.apply(env, down, down.replace(hp=paid_up.hp))
    assert float(reward) == 0.
    _, reward = rules.apply(env, restored, restored.replace(unit_levels=restored.unit_levels.at[hero].add(1)))
    assert float(reward) == 2.
    # A fresh level-1 party after a wipe must exceed the old sum, too.
    wiped, reward = rules.apply(env, paid_up, paid_up.replace(unit_ids=jnp.zeros_like(state.unit_ids),
                                                            hp=jnp.zeros_like(state.hp)))
    assert float(reward) == 0.
    rebuilt, reward = rules.apply(env, wiped, state.replace(party_level_record=wiped.party_level_record))
    assert float(reward) == 0. and int(rebuilt.party_level_record) == int(paid_up.party_level_record)
    # KC0092's capital order, early interception and destruction never earn kill rewards.
    raider = enemy_index(env, 'S143KC0092')
    assert not bool(rules.rewardable[raider])
    bound = state.replace(campaign_targets=state.campaign_targets.at[env.campaign_events.slot[raider]].set(rules.capital_target))
    assert bool(rules.capital_bound(env, bound)[raider])
    dead = bound.replace(alive=bound.alive.at[raider].set(False))
    _, paid = rules.apply(env, bound, dead)
    worth = float(rules.army_xp(env, bound)[raider])
    assert worth > 0 and float(paid) == 0.
    idle = state.replace(campaign_targets=state.campaign_targets.at[env.campaign_events.slot[raider]].set(-1))
    _, unpaid = rules.apply(env, idle, idle.replace(alive=idle.alive.at[raider].set(False)))  # no order
    assert float(unpaid) == 0.


def test_future_capital_attackers_have_no_combat_cast_or_clearing_rewards(mountains):
    env, state = mountains
    rules = env.defense_rewards
    attacker = enemy_index(env, 'S143KC0092')
    ordinary = enemy_index(env, 'S143KC0033')
    scout = enemy_index(env, 'S143TM0000')
    assert rules.suppress_attacker_rewards and not bool(rules.rewardable[attacker])
    assert bool(rules.rewardable[ordinary]) and bool(rules.rewardable[scout])
    # No capital order has fired at reset; early interception is excluded too.
    assert not bool(rules.capital_bound(env, state)[attacker])
    for enemy in (attacker, ordinary):
        before = state.replace(enemy=jnp.int32(enemy), in_battle=jnp.bool_(True),
                               position=jnp.array([12, 20], jnp.int32))
        dead = before.replace(alive=before.alive.at[enemy].set(False), in_battle=jnp.bool_(False))
        scored, reward = reference_rewards.battle(env, before, dead, before.hp,
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))
        assert bool(scored.reference_defeated[enemy]) == (enemy == ordinary)
        assert (float(reward) == 0.) if enemy == attacker else (float(reward) > 4.)
        _, action_reward = reference_rewards.battle(env, before, before, before.hp,
            jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.bool_(True))
        assert float(action_reward) == pytest.approx(0. if enemy == attacker else .0525)
        # A successful damaging/curse cast carries only its target-specific cast bonus.
        cast = state.replace(last_cast_spell=jnp.int32(0), last_spell_target=jnp.int32(enemy))
        _, cast_reward = reference_rewards.world(env, state, cast, jnp.int32(8), jnp.bool_(False))
        assert float(cast_reward) == pytest.approx(0. if enemy == attacker else .005)
    # Even a pre-marked attacker cannot contribute to the final clearing bonus.
    terminal = state.replace(done=jnp.bool_(True), reference_defeated=state.reference_defeated.at[attacker].set(True))
    _, excluded = reference_rewards.finish(env, state, terminal)
    _, empty = reference_rewards.finish(env, state, terminal.replace(reference_defeated=state.reference_defeated))
    assert float(excluded) == float(empty)


def test_day_rewards_require_promotions_at_the_deadline_once(mountains):
    env, state = mountains
    rules = env.defense_rewards
    assert env.map_config['party_day_rewards'] == [dict(day=20, upgrades=4, reward=10.),
        dict(day=50, upgrades=8, reward=20.), dict(day=80, upgrades=12, reward=30.)]
    for index, (day, upgrades, amount) in enumerate(((20, 4, 10.), (50, 8, 20.), (80, 12, 30.))):
        before = state.replace(day=jnp.int32(day-1), party_upgrade_count=jnp.int32(upgrades))
        after = before.replace(day=jnp.int32(day))
        paid, reward = rules.apply(env, before, after)
        assert float(reward) == amount and bool(paid.party_day_milestones[index])
        _, again = rules.apply(env, paid, paid)
        assert float(again) == 0.
        short = before.replace(party_upgrade_count=jnp.int32(upgrades-1))
        missed, reward = rules.apply(env, short, short.replace(day=jnp.int32(day)))
        assert float(reward) == 0. and not bool(missed.party_day_milestones[index])
        # Promotions acquired later on the threshold day or after it do not repair a missed deadline.
        caught_up = missed.replace(party_upgrade_count=jnp.int32(upgrades))
        _, late = rules.apply(env, missed, caught_up)
        assert float(late) == 0.
        _, late = rules.apply(env, caught_up, caught_up.replace(day=jnp.int32(day+1)))
        assert float(late) == 0.
        _, lost = rules.apply(env, before, after.replace(lost=jnp.bool_(True)))
        assert float(lost) == 0.
    assert jax.jit(rules.observation)(state).shape == (rules.observation_size,)


def test_day_promotions_count_repeated_growth_and_exclude_reserves_and_summons(mountains):
    env, state = mountains
    rules = env.defense_rewards
    promoted = jnp.uint32((1 << 0) | (1 << 1) | (1 << 4) | (1 << 6))
    assert int(rules.promotion_count(state, promoted)) == 3  # enemy slot 6 is not the party
    for changes in (dict(garrison_battle=jnp.bool_(True)), dict(summon_battle=jnp.bool_(True)),
                    dict(bot_autobattle=jnp.bool_(True))):
        assert int(rules.promotion_count(state.replace(**changes), promoted)) == 0
    # A second promotion of the same fighter adds another event, regardless of the level record.
    first = state.replace(party_upgrade_count=rules.promotion_count(state, jnp.uint32(2)))
    second = first.replace(party_upgrade_count=first.party_upgrade_count
                           + rules.promotion_count(first, jnp.uint32(2)))
    assert int(second.party_upgrade_count) == 2
    wiped = env.leader_hire.wipe(env, second, jnp.bool_(True))
    assert int(wiped.party_upgrade_count) == 2


def test_party_milestones_multiply_the_entire_episode(mountains):
    env, state = mountains
    rules = env.defense_rewards
    present = state.unit_ids[:6] != 0
    hero = int(jnp.argmax(env.progression.heroes[state.unit_ids[:6]]))
    fighter = next(i for i in range(6) if bool(present[i]) and i != hero)
    all2 = state.replace(unit_levels=state.unit_levels.at[:6].set(jnp.where(present, 2, 0)))
    all3 = state.replace(unit_levels=state.unit_levels.at[:6].set(jnp.where(present, 3, 0)))
    assert float(rules.multiplier(all2)) == 1.5 and float(rules.multiplier(all3)) == 2.
    assert float(rules.multiplier(all2.replace(unit_levels=all2.unit_levels.at[fighter].set(1)))) == 1.
    assert float(rules.multiplier(all2.replace(hp=all2.hp.at[fighter].set(0)))) == 1.
    assert float(rules.multiplier(all3.replace(hp=all3.hp.at[hero].set(0)))) == 1.
    # Garrison troops and battle summons cannot unlock party achievements.
    frozen, bonus = rules.apply(env, state, all3.replace(in_battle=jnp.bool_(True)))
    assert float(bonus) == 0. and float(frozen.party_reward_multiplier) == 1.
    peak, _ = rules.apply(env, state, all3)
    fallen, _ = rules.apply(env, peak, peak.replace(hp=jnp.zeros_like(peak.hp)))
    assert float(fallen.party_reward_multiplier) == 2.
    # Earlier steps, all reward components and the final +10/-10 are included exactly once.
    for multiplier in (1., 1.5, 2.):
        for final_reward in (10., -10.):
            start = state.replace(party_reward_multiplier=jnp.float32(multiplier))
            mid, first = rules.finish(start, start, jnp.float32(3.))
            mid, second = rules.finish(mid, mid, jnp.float32(-1.))
            end, last = rules.finish(mid, mid.replace(done=jnp.bool_(True)), jnp.float32(final_reward))
            assert float(first+second+last) == pytest.approx((2.+final_reward)*multiplier)
            assert float(end.defense_raw_return) == 2.+final_reward
            _, repeated = rules.finish(end, end, jnp.float32(0.))
            assert float(repeated) == 0.


def test_mana_guards_pay_once_separately_from_rods(mountains):
    env, state = mountains
    rules = env.defense_rewards
    guards = sorted(set(env.map_config['rod_planter']['guards']))
    assert rules.guards.tolist() == guards and len(guards) == 5
    guard = guards[0]
    killed = state.replace(alive=state.alive.at[guard].set(False))
    paid, reward = rules.apply(env, state, killed)
    assert float(reward) == .5 and int(jnp.sum(paid.mana_guards_rewarded)) == 1
    assert not bool(jnp.any(paid.mine_rods))  # the +1 for planting is separate
    _, repeated = rules.apply(env, paid, paid)
    assert float(repeated) == 0.
    respawned = paid.replace(alive=paid.alive.at[guard].set(True))
    _, repeated = rules.apply(env, respawned, respawned.replace(alive=paid.alive))
    assert float(repeated) == 0.
    _, reward = rules.apply(env, state, state.replace(alive=state.alive.at[rules.guards].set(False)))
    assert float(reward) == 2.5


def test_personal_field_victories_pay_only_the_living_dukes_party(mountains):
    env, state = mountains
    rules = env.defense_rewards
    hero = int(jnp.argmax(env.progression.heroes[state.unit_ids[:6]]))
    ordinary = enemy_index(env, 'S143KC0033')
    guard = int(rules.guards[0])
    for enemy, amount in ((ordinary, 5.), (guard, 10.)):
        before = state.replace(enemy=jnp.int32(enemy), in_battle=jnp.bool_(True),
                               position=jnp.array([12, 20], jnp.int32))
        after = before.replace(in_battle=jnp.bool_(False), alive=before.alive.at[enemy].set(False))
        paid, bonus = rules.battle(before, after, jnp.bool_(True))
        assert float(bonus) == amount and int(paid.party_field_victories) == 1
        assert bool(paid.defense_defeated[enemy])
        if enemy == guard:
            _, separate = rules.apply(env, before, paid)
            assert float(separate) == .5  # the personal +10 does not replace the general guard reward
        # Even a respawn cannot make this permanent army pay its personal bonus twice.
        again = paid.replace(in_battle=jnp.bool_(True), alive=paid.alive.at[enemy].set(True))
        _, bonus = rules.battle(again, again, jnp.bool_(True))
        assert float(bonus) == 0.
        _, bonus = rules.battle(before, after, jnp.bool_(False))
        assert float(bonus) == 0.
        for changes in (dict(garrison_battle=jnp.bool_(True)), dict(summon_battle=jnp.bool_(True)),
                        dict(bot_autobattle=jnp.bool_(True)), dict(position=rules.capital),
                        dict(enemy_positions=before.enemy_positions.at[enemy].set(rules.capital))):
            _, bonus = rules.battle(before.replace(**changes), after, jnp.bool_(True))
            assert float(bonus) == 0.
        _, bonus = rules.battle(before, after.replace(hp=after.hp.at[hero].set(0)), jnp.bool_(True))
        assert float(bonus) == 0.
    # A created wave is not a permanent army, even when intercepted far from the capital.
    event = enemy_index(env, 'S143TM0007')
    before = state.replace(enemy=jnp.int32(event), position=jnp.array([12, 20], jnp.int32))
    _, bonus = rules.battle(before, before, jnp.bool_(True))
    assert float(bonus) == 0.


def test_whole_party_levels_pay_immediately_once_per_episode(mountains):
    env, state = mountains
    rules = env.defense_rewards
    present = state.unit_ids[:6] != 0
    all2 = state.replace(unit_levels=state.unit_levels.at[:6].set(jnp.where(present, 2, 0)))
    paid2, reward = rules.apply(env, state, all2)
    record_bonus = 2.*int(paid2.party_level_record-state.party_level_record)
    assert float(reward) == record_bonus+10. and paid2.party_level_milestones.tolist() == [True, False]
    all3 = paid2.replace(unit_levels=state.unit_levels.at[:6].set(jnp.where(present, 3, 0)))
    paid3, reward = rules.apply(env, paid2, all3)
    record_bonus = 2.*int(paid3.party_level_record-paid2.party_level_record)
    assert float(reward) == record_bonus+20. and paid3.party_level_milestones.tolist() == [True, True]
    assert not bool(paid3.done)  # these payouts do not wait for the episode's end
    fallen, reward = rules.apply(env, paid3, paid3.replace(hp=jnp.zeros_like(state.hp)))
    assert float(reward) == 0. and bool(jnp.all(fallen.party_level_milestones))
    restored, reward = rules.apply(env, fallen, fallen.replace(hp=state.hp))
    assert float(reward) == 0. and bool(jnp.all(restored.party_level_milestones))
    # First reaching level 3 unlocks both milestones, for +30 in addition to the level record.
    jumped, reward = rules.apply(env, state, all3)
    record_bonus = 2.*int(jumped.party_level_record-state.party_level_record)
    assert float(reward) == record_bonus+30.


def test_repeated_weak_capital_attackers_remain_unrewarded(mountains):
    env, state = mountains
    rules = env.defense_rewards
    enemy = enemy_index(env, 'S143TM0007')  # recurring Ranger, 1378 kill XP
    before = state.replace(enemy=jnp.int32(enemy), alive=state.alive.at[enemy].set(True),
        campaign_targets=state.campaign_targets.at[env.campaign_events.slot[enemy]].set(rules.capital_target))
    worth = float(rules.army_xp(env, before, enemy))
    assert worth == 1378. and bool(rules.repeating[enemy]) and float(rules.repeat_factor(env, before, enemy)) == 1.
    dead = before.replace(alive=before.alive.at[enemy].set(False))
    first, reward = rules.apply(env, before, dead)
    assert float(reward) == 0.
    assert bool(first.defense_defeated[enemy])
    assert int(first.party_field_victories) == 0  # destruction alone is not a personal battle victory
    respawned = first.replace(alive=first.alive.at[enemy].set(True))
    assert float(rules.repeat_factor(env, respawned, enemy)) == pytest.approx(.1)
    _, reward = rules.apply(env, respawned, respawned.replace(alive=dead.alive))
    assert float(reward) == 0.
    # Neither the first nor later copies pay the tactical victory/XP/survival bonus.
    arguments = (jnp.bool_(True), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))
    _, full = reference_rewards.battle(env, before, dead, before.hp, *arguments)
    _, reduced = reference_rewards.battle(env, respawned, dead, before.hp, *arguments)
    assert float(full) == float(reduced) == 0.
    _, lost = reference_rewards.battle(env, respawned, dead, before.hp,
        jnp.bool_(False), jnp.bool_(True), jnp.bool_(False), jnp.bool_(False))
    assert float(lost) == -.25  # losing a repeated battle does not become cheaper
    # A spell/summon killing the first copy also consumes its full-reward occurrence.
    summoned = before.replace(summon_battle=jnp.bool_(True))
    marked, bonus = rules.battle(summoned, dead, jnp.bool_(True))
    assert float(bonus) == 0. and bool(marked.defense_defeated[enemy])
    # Strong recurring opponents and weak nonrecurring armies keep their full payment.
    for source in ('S143TM0024', 'S143TM001E', 'S143TM0028', 'S143KC0092', 'S143TM0017'):
        other = enemy_index(env, source)
        seen = state.replace(defense_defeated=state.defense_defeated.at[other].set(True))
        assert float(rules.repeat_factor(env, seen, other)) == 1.
    # A formerly weak template whose permanent levels grew past the XP threshold is strong now.
    grown = respawned.replace(enemy_progress=respawned.enemy_progress.at[enemy, :, 0].set(
        jnp.where(env.progression.enemy_ids[enemy] != 0, 99, 0)))
    assert float(rules.army_xp(env, grown, enemy)) > rules.repeat_wave_max_xp
    assert float(rules.repeat_factor(env, grown, enemy)) == 1.


def test_destroyed_party_is_not_a_defeat(mountains_rehire):
    env, state = mountains_rehire
    hire = env.leader_hire
    dragons, end = destroyed(env, state.replace(gold=jnp.int32(700)), [30, 30])
    assert not bool(end.in_battle | end.lost | end.done) and bool(end.alive[dragons])
    assert end.unit_ids[:6].tolist() == [0]*6 and end.hp.tolist() == [0]*12
    assert end.position.tolist() == env.map_config['capital_position'] and int(jnp.sum(end.potions)) == 0
    assert int(jnp.sum(end.enemy_wounds[dragons])) > 0 and int(end.gold) == 700
    legal = compiled_method(env, 'action_mask')(end)
    assert not bool(jnp.any(legal[:8])) and bool(legal[REST]) and bool(legal[hire.action])
    rested, _ = compiled_method(env, 'step')(end, jnp.int32(REST))
    assert not bool(rested.done) and not bool(jnp.any(compiled_method(env, 'action_mask')(rested)[:8]))
    hired, _ = compiled_method(env, 'step')(end, jnp.int32(hire.action))
    assert hired.unit_ids[:6].tolist() == [0, int(env.progression.initial_ids[1]), 0, 0, 0, 0]
    assert hired.unit_levels[1] == 1 and int(hired.hp[1]) == 150 and int(hired.gold) == 700-hire.price == 200
    legal = compiled_method(env, 'action_mask')(hired)
    assert bool(jnp.any(legal[:8])) and not bool(legal[hire.action])
    assert env.recruitment.composition(hired)[:3].tolist() == [3, 0, 3]
    poor, _ = compiled_method(env, 'step')(end.replace(gold=jnp.int32(hire.price-1)), jnp.int32(hire.action))
    assert not bool(jnp.any(poor.unit_ids[:6] != 0))


def test_party_destroyed_at_the_capital_leaves_the_garrison(mountains_rehire):
    env, state = mountains_rehire
    home = state.replace(enemy_positions=state.enemy_positions.at[enemy_index(env, 'S143KC0052')].set(
        capital_cell(env)))
    dragons, end = destroyed(env, home, env.map_config['capital_position'])
    assert not bool(end.lost) and bool(end.garrison_pending[dragons])
    legal = compiled_method(env, 'action_mask')(end)
    assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1
    battle, _ = compiled_method(env, 'step')(end, jnp.int32(CONTINUE))
    assert bool(battle.in_battle & battle.garrison_battle) and int(battle.enemy) == dragons
    final, _ = fight(env, battle)
    assert bool(final.lost & final.done) and int(final.garrison_hp[env.garrisons.capital_index, 4]) == 0
    # Without a party an army arriving on REST meets the garrison too.
    goblins = enemy_index(env, 'S143KC0033')
    _, empty = destroyed(env, state, [30, 30])
    empty = empty.replace(enemy_positions=empty.enemy_positions.at[goblins].set(capital_cell(env)))
    attacked, _ = compiled_method(env, 'step')(empty, jnp.int32(REST))
    assert bool(attacked.in_battle & attacked.garrison_battle) and int(attacked.enemy) == goblins


def test_party_destroyed_ends_the_episode_in_the_field_or_capital(mountains):
    env, state = mountains
    rules = env.leader_hire
    assert rules.terminal_party_loss and not rules.can_hire
    assert 'hire_leader' not in action_names(env.map_config)
    assert env.num_actions == len(action_names(env.map_config))
    dragons = enemy_index(env, 'S143KC0052')
    for position in ([30, 30], env.map_config['capital_position']):
        before = state.replace(enemy_positions=state.enemy_positions.at[dragons].set(jnp.array(position)))
        _, end = destroyed(env, before, position)
        assert bool(end.done & end.lost) and not bool(end.won | end.in_battle)
        assert int(end.party_losses) == 1
        assert bool(end.alive[dragons]) and not bool(end.garrison_pending[dragons])
        assert not bool(rules.available(end.replace(gold=jnp.int32(99999))))
        legal = compiled_method(env, 'action_mask')(end)
        assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1  # terminal placeholder only
        still_done, _ = compiled_method(env, 'step')(end, jnp.int32(CONTINUE))
        assert bool(still_done.lost & still_done.done) and int(still_done.party_losses) == 1
        assert int(end.capital_captor) == -1  # party loss is not a captured capital
    _, party_penalty = reference_rewards.finish(env, state, end)
    _, capital_penalty = reference_rewards.finish(env, state,
        state.replace(lost=jnp.bool_(True), done=jnp.bool_(True)))
    assert float(party_penalty) == float(capital_penalty) == -10.


def test_event_spell_destroying_the_party_is_a_terminal_defeat(mountains):
    env, state = mountains
    killed = jax.jit(lambda s: env.campaign_events._cast(
        env, s, jnp.bool_(True), dict(amount=99999, bit=0)))(state)
    assert bool(killed.done & killed.lost) and not bool(killed.won)
    assert not bool(jnp.any(killed.hp[:6] > 0)) and int(killed.party_losses) == 1
    legal = compiled_method(env, 'action_mask')(killed)
    assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1
    repeated = env.campaign_events._cast(env, killed, jnp.bool_(True), dict(amount=99999, bit=0))
    assert bool(repeated.lost & repeated.done) and int(repeated.party_losses) == 1


def test_party_inside_and_waiting_battles(mountains):
    env, state = mountains
    enemy = enemy_index(env, 'S143KC0033')
    home = state.replace(enemy_positions=state.enemy_positions.at[enemy].set(capital_cell(env)))
    advanced = jax.jit(lambda s: env.garrisons.advance(s))(home)
    assert not bool(jnp.any(advanced.garrison_pending))  # the party defends first
    waiting = state.replace(garrison_pending=state.garrison_pending.at[enemy].set(True))
    legal = compiled_method(env, 'action_mask')(waiting)
    assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1


def test_original_victory_after_shornhelm(mountains):
    env, state = mountains
    won = compiled_method(env, '_campaign_won')
    step = compiled_method(env, 'step')
    goal = next(i for i, c in enumerate(env.map_config['cities']) if c['objective'])
    allemon, gumtik = enemy_index(env, 'S143TM0014'), enemy_index(env, 'S143TM0015')
    assert not bool(won(state))
    entrance = jnp.array(env.map_config['cities'][goal]['position'], jnp.int32)
    defenders = jnp.all(state.enemy_positions == entrance, axis=1)  # the capture killed them
    taken = state.replace(city_owned=state.city_owned.at[goal].set(True), alive=state.alive & ~defenders)
    assert not bool(won(taken))  # holding Shornhelm alone is not the victory
    after, _ = step(taken, jnp.int32(REST))
    slot = env.campaign_events.slot
    assert bool(after.alive[allemon] & after.alive[gumtik]) and not bool(after.won)
    targets = env.campaign_events.targets
    assert targets[after.campaign_targets[slot[allemon]]].tolist() == env.map_config['capital_position']
    assert targets[after.campaign_targets[slot[gumtik]]].tolist() == env.map_config['cities'][goal]['position']
    one = after.replace(alive=after.alive.at[allemon].set(False))
    assert not bool(won(one)) and bool(won(one.replace(alive=one.alive.at[gumtik].set(False))))
    lost = one.replace(alive=one.alive.at[gumtik].set(False), city_owned=one.city_owned.at[goal].set(False))
    assert not bool(won(lost))


def test_events_compile_the_legions_side():
    config = load_map('wild_mountains')
    ids, events = config['reference_enemy_ids'], config['campaign_events']['events']
    assert len(events) == 40 and len(config['campaign_events']['armies']) == 40
    assert ids[:164] == sorted(ids[:164]) and all(i.startswith('S143TM') for i in ids[164:])
    assert config['scenario_event_enemies'] == list(range(164, 190))
    names = {e['id'] for e in events}
    assert {'S143EV0002', 'S143EV0005', 'S143EV0018', 'S143EV0019', 'S143EV0021', 'S143EV0023'}.isdisjoint(names)
    capital = config['capital_position']
    sentinel = next(c['position'] for c in config['enemy_capitals'])
    targets = [f.get('target') for e in events for f in e['effects']]
    assert capital in targets and sentinel not in targets
    scouts = next(e for e in events if e['id'] == 'S143EV0003')
    assert not scouts['once'] and scouts['conditions'] == [dict(kind='FREQUENCY', days=6)]


def test_scout_wave_reaches_the_capital_by_the_shortest_path(mountains):
    """The first eastern scout appears on day 6, walks to the capital zone and sends KC0092."""
    env, state = mountains
    step = compiled_method(env, 'step')
    scout, raider = enemy_index(env, 'S143TM0000'), enemy_index(env, 'S143KC0092')
    rules = env.campaign_events
    s = state
    for _ in range(5):
        s, _ = step(s, jnp.int32(REST))
    assert int(s.day) == 6 and bool(s.alive[scout])
    target = rules.targets[s.campaign_targets[rules.slot[scout]]]
    distance = rules.distances[s.campaign_targets[rules.slot[scout]]]
    before = int(distance[s.enemy_positions[scout][0]*env.size+s.enemy_positions[scout][1]])
    s, _ = step(s, jnp.int32(REST))
    after = int(distance[s.enemy_positions[scout][0]*env.size+s.enemy_positions[scout][1]])
    assert after < before and not bool(env.territory.blocked(s.enemy_positions[scout][None])[0])
    for _ in range(12):
        if bool(s.campaign_fired[[e['id'] for e in env.map_config['campaign_events']['events']].index('S143EV0004')]):
            break
        s, _ = step(s, jnp.int32(REST))
    assert int(jnp.max(jnp.abs(s.enemy_positions[scout]-target))) <= 1
    assert rules.targets[s.campaign_targets[rules.slot[raider]]].tolist() == env.map_config['capital_position']


def test_large_mercenary_takes_a_column(mountains):
    env, state = mountains
    rules = env.recruitment
    offer = rules.keys.index('g000uu5023')  # Green Dragon
    camp = env.map_config['mercenaries'][0]['position']
    alone = state.replace(unit_ids=state.unit_ids.at[jnp.array([0, 4])].set(0),
                          hp=state.hp.at[jnp.array([0, 4])].set(0), gold=jnp.int32(10000),
                          position=jnp.array([camp[0]+3, camp[1]+3], jnp.int32))
    assert bool(rules.available(alone)[offer])
    after, _ = compiled_method(env, 'step')(alone, jnp.int32(rules.start+offer))
    slot = int(rules.targets(alone)[offer])
    assert int(after.unit_ids[slot]) == int(rules.ids[offer]) and int(after.unit_ids[slot+3]) == 0
    assert rules.composition(after)[1].tolist() == 2


def test_baroness_rods_pay_like_land(mountains):
    env, state = mountains
    rods, land = env.rods, env.territory
    step = compiled_method(env, 'step')
    mask = compiled_method(env, 'action_mask')
    config = env.map_config['rod_planter']
    # The five mana sources nearest by walking, off the starting land, each with its nearest army.
    assert config['sources'] == [0, 14, 9, 8, 6] and (rods.cost, rods.rod_cost, rods.reward) == (300, 150, 1.)
    ids = env.map_config['reference_enemy_ids']
    assert [ids[g] for g in config['guards']] == ['S143KC0013', 'S143KC005b', 'S143KC0045', 'S143KC0044', 'S143KC0086']
    assert all(env.map_config['mines'][k]['kind'] != 'gold' for k in config['sources'])
    assert not bool(jnp.any(mask(state.replace(gold=jnp.int32(9999)))[rods.plant_start:rods.end]))  # no Baroness
    hired, _ = step(state.replace(gold=jnp.int32(1000)), jnp.int32(rods.start))
    assert bool(hired.rod_planter) and int(hired.gold) == 700 and not bool(mask(hired)[rods.start])
    assert not bool(jnp.any(mask(hired)[rods.plant_start:rods.end]))  # every guard still stands
    freed = hired.replace(alive=hired.alive.at[config['guards'][0]].set(False))
    assert mask(freed)[rods.plant_start:rods.end].tolist() == [True, False, False, False, False]
    planted, _ = step(freed, jnp.int32(rods.plant_start))
    death = 3  # resource index: gold, infernal, life, death, runes, elves
    assert bool(planted.mine_rods[0]) and int(planted.gold) == 550 and not bool(mask(planted)[rods.plant_start])
    assert int(land.income(planted)[death]-land.income(freed)[death]) == 50
    assert jnp.array_equal(planted.territory_owners, freed.territory_owners)  # the land stays


def test_thief_poisons_the_next_capital_attacker():
    plain = NumberGrid(map_config=load_map('wild_mountains'))
    state, _ = compiled_method(plain, 'reset')(jax.random.PRNGKey(143))
    assert plain.thief.enabled and not bool(compiled_method(plain, 'action_mask')(state)[plain.thief.action])
    # An artificial wave: the goblins appear on the first REST and reach the capital on the second.
    game_map = load_map('wild_mountains')
    goblins = game_map['reference_enemy_ids'].index('S143KC0033')
    game_map['scheduled_enemies'] = [dict(enemy=goblins, moves_per_turn=4, spawn_turn=0,
                                          target=game_map['capital_position'])]
    env = NumberGrid(map_config=game_map)
    step, mask = compiled_method(env, 'step'), compiled_method(env, 'action_mask')
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    assert not bool(mask(state)[env.thief.action])  # not on the map yet
    day2, _ = step(state, jnp.int32(REST))
    assert bool(day2.alive[goblins]) and bool(mask(day2)[env.thief.action])
    assert not bool(day2.thief_alive)
    assert not bool(mask(day2.replace(gold=jnp.int32(199)))[env.thief.action])  # no thief, no gold
    assert bool(mask(day2.replace(gold=jnp.int32(200)))[env.thief.action])  # the first hire
    day2 = day2.replace(gold=jnp.int32(1000))
    events = {}
    for seed in range(8):
        tried, _ = step(day2.replace(battle_key=jax.random.PRNGKey(seed)), jnp.int32(env.thief.action))
        assert int(tried.gold) == int(day2.gold)-200 and int(tried.thief_attempts_today) == 1
        events.setdefault(int(tried.last_event), tried)
    poisoned, failed = events[THIEF_POISONED], events[THIEF_FAILED]
    assert not bool(jnp.any(failed.enemy_wounds[goblins])) and not bool(failed.thief_alive)
    assert bool(mask(failed)[env.thief.action])  # gold permits another hire on the same day
    for gold in (0, 199, 200):
        assert not bool(mask(failed.replace(gold=jnp.int32(gold)))[env.thief.action])
    assert bool(mask(failed.replace(gold=jnp.int32(201)))[env.thief.action])
    # Goblins 50 HP -> 40, goblin archers 40 -> 32: 20% of current HP, truncated.
    assert poisoned.enemy_wounds[goblins].tolist() == [10, 10, 10, 8, 0, 8] and bool(poisoned.thief_alive)
    # User rule 09.11.2026: a thief who poisoned the army may try once more the same day, for free.
    assert bool(mask(poisoned.replace(gold=jnp.int32(0)))[env.thief.action])
    second, _ = step(poisoned.replace(gold=jnp.int32(0)), jnp.int32(env.thief.action))
    assert int(second.gold) == 0 and int(second.thief_attempts_today) == 2
    assert int(second.last_event) in (THIEF_POISONED, THIEF_FAILED) and not bool(mask(second)[env.thief.action])
    if int(second.last_event) == THIEF_POISONED:  # 40 -> 32, 32 -> 26 (6.4 truncated to 6)
        assert second.enemy_wounds[goblins].tolist() == [18, 18, 18, 14, 0, 14]

    def try_for(before, event):
        for seed in range(32):
            after, _ = step(before.replace(battle_key=jax.random.PRNGKey(seed)), jnp.int32(env.thief.action))
            if int(after.last_event) == event:
                return after
        raise AssertionError(f'No thief outcome {event} found')

    # Replace the failed thief, use his free success, then replace him alive and repeat.
    # More than two total attempts are allowed; only each paid hire resets the two-try limit.
    replacement = try_for(failed.replace(gold=jnp.int32(401)), THIEF_POISONED)
    assert int(replacement.gold) == 201 and int(replacement.last_service_cost) == 200
    assert int(replacement.thief_attempts_today) == 1
    free = try_for(replacement, THIEF_POISONED)
    assert int(free.gold) == 201 and int(free.last_service_cost) == 0
    assert int(free.thief_attempts_today) == 2 and bool(free.thief_alive)
    assert not bool(mask(free.replace(gold=jnp.int32(200)))[env.thief.action])
    replacement = try_for(free, THIEF_POISONED)
    assert int(replacement.gold) == 1 and int(replacement.last_service_cost) == 200
    assert int(replacement.thief_attempts_today) == 1 and bool(mask(replacement)[env.thief.action])
    exhausted = try_for(replacement, THIEF_POISONED)
    assert int(exhausted.gold) == 1 and int(exhausted.last_service_cost) == 0
    assert int(exhausted.thief_attempts_today) == 2 and not bool(mask(exhausted)[env.thief.action])
    # Failure on the free repeat also permits a paid replacement, with a fresh first try.
    second_failed = try_for(poisoned, THIEF_FAILED)
    assert int(second_failed.gold) == int(poisoned.gold) and not bool(second_failed.thief_alive)
    assert int(second_failed.thief_attempts_today) == 2 and bool(mask(second_failed)[env.thief.action])
    paid, _ = step(second_failed, jnp.int32(env.thief.action))
    assert int(paid.gold) == int(second_failed.gold)-200 and int(paid.thief_attempts_today) == 1
    # The next day a living thief's tries reset; a dead one needs gold for the day's first hire.
    assert not bool(mask(failed.replace(thief_attempts_today=jnp.int32(0), gold=jnp.int32(199)))[env.thief.action])
    assert bool(mask(failed.replace(thief_attempts_today=jnp.int32(0), gold=jnp.int32(200)))[env.thief.action])
    assert bool(mask(exhausted.replace(thief_attempts_today=jnp.int32(0), gold=jnp.int32(0)))[env.thief.action])
    day3, _ = step(poisoned, jnp.int32(REST))
    assert int(day3.thief_attempts_today) == 0 and bool(day3.in_battle) and int(day3.enemy) == goblins


def test_spells_strike_the_next_capital_attacker():
    """Damage, curses and summons may target the army the thief would poison, one turn before its attack."""
    game_map = load_map('wild_mountains')
    goblins = game_map['reference_enemy_ids'].index('S143KC0033')
    game_map['scheduled_enemies'] = [dict(enemy=goblins, moves_per_turn=4, spawn_turn=0,
                                          target=game_map['capital_position'])]
    env = NumberGrid(map_config=game_map)
    rules, casting, names = env.attacker_spells, env.casting, action_names(game_map)
    assert rules.enabled and rules.count and all(
        names[rules.start+j] == names[int(base)]+'_at_attacker' for j, base in enumerate(rules.bases.tolist()))
    kinds = {int(casting.kinds[int(base)-casting.start if int(base) < casting.hire_action
                               else int(casting.item_rows[int(base)-casting.item_start])]) for base in rules.bases}
    assert kinds == {DAMAGE_SPELL, DEBUFF, SUMMON_SPELL}  # curses last through the enemy turn
    step, mask = compiled_method(env, 'step'), compiled_method(env, 'action_mask')
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    def ready(s):
        return s.replace(mana=jnp.full_like(s.mana, 9999), learned_spells=jnp.full_like(s.learned_spells, 0xffffffff))
    assert not bool(jnp.any(mask(ready(state))[rules.start:rules.end]))  # not on the map yet
    day2, _ = step(state, jnp.int32(REST))
    day2 = ready(day2)
    wisp = rules.start + [int(b) for b in rules.bases].index(casting.start + casting_row(casting, 'g000ss0043'))
    nearest = int(casting.targets(day2)[0])
    assert bool(casting.field[goblins]) and bool(mask(day2)[wisp])  # damage needs a field army
    struck, _ = step(day2, jnp.int32(wisp))
    assert int(struck.last_spell_target) == goblins and int(struck.last_cast_spell) == casting_row(casting, 'g000ss0043')
    assert int(jnp.sum(struck.enemy_wounds[goblins])) > 0 and int(jnp.sum(struck.mana)) < int(jnp.sum(day2.mana))
    normal, _ = step(day2, jnp.int32(casting.start + casting_row(casting, 'g000ss0043')))
    assert int(normal.last_spell_target) == nearest
    hound = rules.start + [int(b) for b in rules.bases].index(casting.start + casting_row(casting, 'g000ss0041'))
    summoned, _ = step(day2, jnp.int32(hound))
    assert int(summoned.last_spell_target) == goblins and int(summoned.last_cast_spell) == casting_row(casting, 'g000ss0041')
    # Without that army every such action is masked and refused.
    calm = day2.replace(alive=day2.alive.at[goblins].set(False))
    assert not bool(jnp.any(mask(calm)[rules.start:rules.end]))
    assert int(step(calm, jnp.int32(wisp))[0].last_cast_spell) < 0


def test_curse_lasts_through_the_enemy_turn_attack():
    """User rule 09.11.2026: a curse on an army attacking the capital in the enemy turn lasts in that battle."""
    game_map = load_map('wild_mountains')
    goblins = game_map['reference_enemy_ids'].index('S143KC0033')
    game_map['scheduled_enemies'] = [dict(enemy=goblins, moves_per_turn=4, spawn_turn=0,
                                          target=game_map['capital_position'])]
    env = NumberGrid(map_config=game_map)
    rules, casting = env.attacker_spells, env.casting
    step = compiled_method(env, 'step')
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    day2, _ = step(state, jnp.int32(REST))
    day2 = day2.replace(mana=jnp.full_like(day2.mana, 9999), learned_spells=jnp.full_like(day2.learned_spells, 0xffffffff))
    torment = casting_row(casting, 'g000ss0045')  # Адские муки: armour -50
    curse = rules.start + [int(b) for b in rules.bases].index(casting.start + torment)
    other = enemy_index(env, 'S143KC0017')
    cursed, _ = step(day2.replace(enemy_spell_effects=day2.enemy_spell_effects.at[other].set(casting.bits[torment])),
                     jnp.int32(curse))
    assert bool(casting.effect_active(cursed.enemy_spell_effects[goblins], casting.bits[torment]))
    night, _ = step(cursed, jnp.int32(REST))  # the party stands in the capital: it fights first
    assert bool(night.in_battle) and int(night.enemy) == goblins
    assert bool(night.enemy_curse_carried[goblins]) and float(night.spell_mods[6+jnp.argmax(night.unit_ids[6:] != 0), 3]) == -50.
    assert not bool(jnp.any(night.enemy_spell_effects[other] != 0))  # no attack: the curse ended on REST
    end, _ = fight(env, night)
    assert not bool(end.enemy_curse_carried[goblins]) and not bool(jnp.any(end.enemy_spell_effects[goblins] != 0))


def casting_row(casting, spell):
    return [row['id'] for row in casting.rows].index(spell)
