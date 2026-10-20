"""Share the immutable current map and compiled GPU transitions between mechanics suites."""
from stoix.tests.number_grid_fixtures import compiled_method
import copy
import jax
import jax.numpy as jnp
from stoix.envs.number_grid import NumberGrid, MAP
import pytest

from stoix.tests.number_grid_fixtures import current_environment


@pytest.fixture(scope='session')
def current_game():
    env = current_environment()
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    return env, state, compiled_method(env,'step'), compiled_method(env,'_battle_step')


@pytest.fixture(scope='session')
def default_game():
    """The Python reference Default map; its battle-item suite shares these programs."""
    from stoix.envs.number_grid import load_map
    env = NumberGrid(map_config=load_map('default'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state, compiled_method(env, 'step'), compiled_method(env, 'action_mask')


@pytest.fixture(scope='session')
def human_service(current_game):
    from collections import OrderedDict
    import threading
    from serve_number_grid import GameService
    env, _, step, _ = current_game
    # Serialize the real accessors through the same cached GPU functions as the
    # mechanics tests. Eager snapshot assembly otherwise compiles hundreds of
    # tiny kernels again for host arrays returned by device_get.
    view = copy.copy(env)
    for name in ('action_mask', 'map_commands', 'max_hp', 'cast_quotes',
                 'round_limit_retreat', 'rest_penalty', 'movement_cap',
                 'travel_costs', 'territory_quotes', 'unit_experience', 'unit_stats'):
        setattr(view, name, compiled_method(env, name))
    for name, method in (('construction', 'status'), ('spell_research', 'status'),
                         ('capital', 'quotes'), ('potion_rules', 'quotes'),
                         ('recruitment', 'quotes'), ('site_rules', 'quotes')):
        rules = getattr(env, name)
        prepared = copy.copy(rules)
        setattr(prepared, method, compiled_method(rules, method))
        setattr(view, name, prepared)
    service = GameService.__new__(GameService)
    service.env = view
    service.environments = {(env.construction.faction, env.construction.lord['id']):(view,compiled_method(env,'reset'),step)}
    service.sessions = OrderedDict()
    service.lock = threading.Lock()
    return service


@pytest.fixture(scope='session')
def training_autoreset(request):
    from numbergrid_config import make_config
    from stoix.utils.make_env import make
    # The wrapper contract needs a real combat environment, but not another
    # compile of every unit effect. The full current-map case stays addressable.
    from stoix.tests.number_grid_fixtures import MAP as BASIC_MAP
    config = make_config(map_config=BASIC_MAP if getattr(request, 'param', None) == 'contract' else None)
    config.env.kwargs.max_steps = 1
    env, eval_env = make(config)
    return env, eval_env, compiled_method(env,'step')


@pytest.fixture(scope='session')
def elemental_attack_game():
    config = copy.deepcopy(MAP)
    config['fear_paralysis_teams'] = ['red']
    config['lord_type'] = 'warrior'
    config['hero_roster'] = ['possessed','duke','possessed','sentry','watcher',None]
    config['hero_units'] = 5
    config['hero_combat_stats'] = [{},{},{},dict(attack_type='weapon'),dict(attack_type='weapon')]
    config['enemy_rosters'][40] = ['succubus','succubus','archer','archer','archer','archer']
    config['enemy_units'][40] = 6
    config['enemy_rosters'][39] = ['uter_betrezen','uter_betrezen','archer','archer','archer','archer']
    config['enemy_units'][39] = 6
    config['enemy_rosters'][38] = ['abbess','prophetess','cleric','elf_oracle','archer','shamanka']
    config['enemy_units'][38] = 6
    # Immutable synthetic profiles shared by Mage and Witch regressions.
    config['enemy_rosters'][0] = ['possessed','possessed','apprentice','archer','archer','archer']
    config['enemy_units'][0] = 6
    config['enemy_rosters'][1] = ['teurg',None,None,None,None,None]
    config['enemy_units'][1] = 1
    config['enemy_rosters'][37] = ['squire','squire','squire','archer','archer','archer']
    config['enemy_units'][37] = 6
    config['enemy_rosters'][2] = ['archer']*6
    config['enemy_units'][2] = 6
    config['enemy_rosters'][36] = list(MAP['enemy_rosters'][11])
    config['enemy_units'][36] = 6
    overrides = [[{} for _ in range(n)] for n in config['enemy_units']]
    overrides[0] = [dict(immunities=['weapon','mind']),
        dict(immunities=['weapon'],protections=['mind']),dict(hero=True),
        dict(immunities=['air']),dict(protections=['air']),{}]
    overrides[1] = [dict(secondary_attack_type='water',secondary_accuracy=0)]
    overrides[37] = [dict(max_hp=100,armor=30,**extra) for extra in
        (dict(immunities=['water']),dict(immunities=['air']),dict(protections=['water']),
         dict(protections=['air']),{},dict(initiative=49))]
    overrides[39] = [dict(secondary_attack_type=''),dict(secondary_accuracy=0),{},{},{},{}]
    overrides[22] = [dict(max_hp=1000,**e) for e in [dict(protections=['mind']),
        dict(immunities=['mind']),dict(protections=['fire']),dict(immunities=['fire']),{},{}]]
    overrides[40] = [dict(attack_type='fire',secondary_attack_type='mind',accuracy=100),
        dict(attack_type='fire',secondary_attack_type='',accuracy=100),
        dict(max_hp=999,protections=['mind']),dict(immunities=['fire']),
        dict(protections=['mind']),dict(immunities=['mind'])]
    extras = [dict(protections=['water','fire','mind','earth']),dict(immunities=['water','fire','mind','earth']),
              dict(immunities=['weapon']),dict(immunities=['poison']),{},{}]
    overrides[11] = [dict(max_hp=300,**e) for e in extras]
    overrides[2] = [dict(immunities=['FiRe'],protections=['fire']),dict(protections=['FIRE','fire'])]+[{}]*4
    overrides[36] = [dict(max_hp=300,**e) for e in
        (dict(protections=['death']),dict(immunities=['death']),dict(immunities=['weapon']),
         dict(immunities=['poison']),{}, {})]
    config['enemy_combat_stats'] = overrides
    env = NumberGrid(map_config=config)
    env.progression.capital_guards = env.progression.capital_guards.at[env.progression.enemy_ids[40,2]].set(True)
    env.progression.capital_guards = env.progression.capital_guards.at[env.progression.enemy_ids[37,5]].set(True)
    initial,_ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    start = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    return env,initial,start.replace(priority=start.priority.at[0].set(10000.))
