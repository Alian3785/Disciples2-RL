"""CUDA contracts: the Python reference Default map, its spell shop and capital guard."""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, NumberGrid, REST, WAIT, action_names, load_map
from stoix.envs.number_grid_battle_items import battle_item_action_names
from stoix.envs.number_grid_combat import DAMAGE
from stoix.envs.number_grid_lords import STARTING_PARTIES
from stoix.envs.number_grid_casting import SPELL_CAST, SPELL_KILL
from stoix.envs.number_grid_progression import ProgressionRules
from stoix.envs.number_grid_sites import BOUGHT, SPELL_BOUGHT, merchant_items
from stoix.envs.number_grid_territory import CAPITAL_ARMOR
from stoix.envs.number_grid_unit_swaps import unit_swap_action_names
from stoix.tests.number_grid_fixtures import compiled_method


pytestmark = pytest.mark.core

DEFAULT = load_map('default')
SHOP = (4, 46)  # spell shop at [3, 43], approach offset (1, 3)
# Twin Island Mages in the user's saves of the original: Ice Pillar, Plague, Stone Rain, Healing.
TOWER = ('g000ss0028', 'g000ss0067', 'g000ss0070', 'g000ss0029')
PLAGUE = 'g000ss0067'
GUARD, DRAGON = 74, 30  # reference enemy ids 75 (Myzrael) and 31 (Green Dragon)
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def test_default_map_layout_and_actions(default_game):
    env, state, _, _ = default_game
    names = action_names(DEFAULT)
    assert DEFAULT['size'] == 50 and len(DEFAULT['opponent_positions']) == 75
    assert env.num_actions == len(names) == 306 and env.observation_size == 2661
    assert env.observation_version == 44 and state.item_charges_spent.shape == state.item_inventory.shape
    assert [s['kind'] for s in env.site_rules.sites] == [
        'merchant', 'merchant', 'trainer', 'mercenary', 'mercenary', 'spell_shop']
    assert env.territory.count == 5 and len(env.territory.mines) == 12 and env.ruins.count == 5
    # Original .sg: 17 bags; the Zombie Talisman comes only from Соругирилла's vault.
    assert len(DEFAULT['chests']) == 17 and env.item_rules.capacity == 14+1+4  # loot + Alar's Runestone + city items
    assert not any('zombie_talisman' in c['items'] for c in DEFAULT['chests'])
    # Same file: five Undead Warlocks (death attack), not the Legions Sorcerer ('warlock' key, fire).
    assert sum('reference_g000uu0081' in r for r in DEFAULT['enemy_rosters']) == 5
    assert not any('warlock' in r for r in DEFAULT['enemy_rosters'])
    assert env.progression.rows[env.progression.ids['reference_g000uu0081']]['attack_type'] == 'death'
    shop = env.site_rules
    assert names[shop.spell_start:shop.end] == tuple('buy_spell_'+spell for spell in TOWER)
    # Legions book (17), the four tower spells, then the Empire ward and healing known only from items.
    assert names[env.casting.start:env.casting.hire_action][-1] == 'cast_'+TOWER[-1] and env.casting.count == 23
    assert names[env.casting.hire_action:env.casting.end] == (
        'hire_scroll_mage', 'cast_mind_ward_scroll', 'cast_staff_of_holiness')  # staff: Кордилия's vault
    assert names[env.casting.end:env.unit_swaps.start] == battle_item_action_names(DEFAULT)  # docs/BATTLE_ITEMS.md
    assert names[env.unit_swaps.start:] == unit_swap_action_names(env.map_config)  # docs/UNIT_SWAPS.md
    # Alar sells the reference potions, then the requested artifact; Tralar the original potions.
    alar, tralar = DEFAULT['merchants']
    assert names[shop.buy_start:shop.train_start] == tuple(
        'buy_0_'+k for k in [*alar['potions'], *alar['items']]) + tuple('buy_1_'+k for k in tralar['potions'])
    assert tralar['potions'] == {'life': 10, 'healing': 10, 'restoration': 10, 'ointment': 10,
                                 'accuracy': 1, 'swiftness': 1, 'vigor': 1, 'bark': 1}
    # Lytharos Camp: Barbarian Warrior; Erarick's Camp: a Wolf (original saves).
    assert [int(i) for i in env.recruitment.offer_sites] == [3, 4]
    assert [n for n in names if n.startswith('hire_mercenary_')] == [
        'hire_mercenary_g000uu5040', 'hire_mercenary_g000uu5039']
    assert DEFAULT['enemy_rosters'][GUARD] == [None]*4+['g000uu3001', None]
    assert DEFAULT['enemy_capitals'][0]['position'] == DEFAULT['opponent_positions'][GUARD] == [11, 27]
    assert [int(i) for i in jnp.flatnonzero(env.territory.capital_guards)] == [GUARD]
    rows = env.progression.rows
    # No fixed roster: the Legions warrior ruler's party (docs/STARTING_PARTIES.md).
    assert DEFAULT['faction'] == 'legions' and DEFAULT['lord_type'] == 'warrior'
    assert 'hero_roster' not in DEFAULT and env.map_config['hero_roster'] == STARTING_PARTIES['legions']['warrior']
    assert [rows[i]['key'] if i else None for i in state.unit_ids[:6].tolist()] == env.map_config['hero_roster']
    assert state.position.tolist() == [28, 6]
    assert int(state.gold) == 0 and state.spell_stock.tolist() == [1, 1, 1, 1]


def test_alar_sells_an_artifact(default_game):
    """User request: one artifact in Alar's shop; the merchant item mechanism is generic."""
    env, initial, step, mask = default_game
    shop, items, names = env.site_rules, env.item_rules, action_names(DEFAULT)
    assert DEFAULT['merchants'][0]['items'] == {'runestone': 1}
    buy = names.index('buy_0_runestone')  # Alar is merchant 0; Tralar sells too
    offer = buy-shop.buy_start
    assert offer == 10 and shop.metadata['buy'][offer]['price'] == 500  # GItem.VALUE
    alar, tralar = (jnp.array(site['interaction_tiles'][0], jnp.int32) for site in shop.sites[:2])
    rich = initial.replace(gold=jnp.int32(600))
    assert mask(rich.replace(position=alar))[buy]
    assert not mask(rich)[buy] and not mask(rich.replace(position=tralar))[buy]
    assert not mask(rich.replace(position=alar, gold=jnp.int32(499)))[buy]
    # The warrior wears artifacts at once: the purchase is equipped in the same step.
    bought, _ = step(rich.replace(position=alar), jnp.int32(buy))
    stone = items.keys.index('runestone')
    assert bought.last_event == BOUGHT and bought.gold == 100 and bought.merchant_stock[offer] == 0
    assert bought.item_inventory[0] == stone and bought.equipped[0] == stone
    assert not mask(bought.replace(gold=jnp.int32(600)))[buy]  # one copy
    again, _ = step(bought.replace(gold=jnp.int32(600)), jnp.int32(buy))
    assert again.gold == 600 and again.item_inventory.tolist() == bought.item_inventory.tolist()
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(1))
    assert reset.merchant_stock.tolist() == shop.initial_stock.tolist()


def test_merchants_sell_any_priced_item_but_valuables():
    stock = {'bethrezens_claw': 1, 'tome_war': 2, 'zombie_talisman': 1, 'lightning_scroll': 1,
             'staff_of_thunder': 1}
    assert merchant_items({'items': stock}) == stock
    for stock in ({'bronze_ring': 1}, {'orb_of_lycanthropy': 1}, {'runestone': 0}, {'unknown': 1}):
        with pytest.raises(ValueError):
            merchant_items({'items': stock})


def test_spell_shop_sells_the_original_tower_spells(default_game):
    """Installed game (the user's saves, turn 11): Twin Island Mages sell four level II spells, 400 each."""
    env, initial, step, mask = default_game
    shop, casting = env.site_rules, env.casting
    ids = [row['id'] for row in casting.rows]
    assert [o['price'] for o in shop.spell_offers] == [400]*4  # Gspells BUY_C
    plague = ids.index(PLAGUE)
    buy = jnp.int32(shop.spell_start+TOWER.index(PLAGUE))
    at_shop = initial.replace(position=jnp.array(SHOP, jnp.int32), gold=jnp.int32(400))
    assert jnp.all(mask(at_shop)[shop.spell_start:shop.end])
    assert not mask(at_shop.replace(gold=jnp.int32(399)))[buy]
    assert not mask(initial.replace(gold=jnp.int32(400)))[buy]
    bought, _ = step(at_shop, buy)
    assert bought.last_event == SPELL_BOUGHT and bought.gold == 0
    assert bought.spell_stock.tolist() == [1, 0, 1, 1] and bought.learned_spells == 1 << plague
    assert not jnp.any(mask(bought)[shop.spell_start:shop.end])  # one copy, and 0 < 400 gold
    # Plague is not in the Legions book, so no research; the bought copy costs 200 death mana.
    cast = casting.start+plague
    assert not mask(bought)[cast]
    armed = bought.replace(mana=jnp.array([0, 0, 200, 0, 0], jnp.int32))
    assert mask(armed)[cast]
    struck, _ = step(armed, jnp.int32(cast))
    assert struck.last_event in (SPELL_CAST, SPELL_KILL) and not jnp.any(struck.mana)
    assert struck.last_spell_target >= 0 and casting.field[struck.last_spell_target]


def test_capital_guard_armor_and_regeneration(default_game):
    from stoix.envs.number_grid import FORT_RED, RETREAT
    env, initial, step, _ = default_game
    guard = DEFAULT['opponent_positions'][GUARD]
    assert not env.casting.field[GUARD]  # damage and weakening never target the capital
    blocked = env.territory.blocked_grid.reshape(50, 50)
    direction, (dr, dc) = next((i, d) for i, d in enumerate(DIRECTIONS)
                               if not blocked[guard[0]-d[0], guard[1]-d[1]])
    # Mizrael (initiative 90) opens with 250 Life damage to the whole party at 95% accuracy,
    # which wipes out the level-1 Legions party; enough HP keeps the battle open to inspect.
    near = initial.replace(position=jnp.array([guard[0]-dr, guard[1]-dc], jnp.int32),
                           hp=initial.hp.at[:6].set(jnp.where(initial.hp[:6] > 0, 1000, 0)))
    battle, _ = step(near, jnp.int32(direction))
    assert battle.in_battle and battle.enemy == GUARD and battle.battle_fort_armor == CAPITAL_ARMOR
    assert battle.battle_fort_defender == FORT_RED
    assert compiled_method(env,'action_mask')(battle.replace(actor=jnp.int32(1)))[RETREAT]
    visitor = initial.replace(enemy=jnp.int32(0),
        enemy_positions=initial.enemy_positions.at[0].set(jnp.array(guard,jnp.int32)))
    assert compiled_method(env.territory,'defender_in_fort')(visitor)
    # Rest: Myzrael's own 50% +35% capital of 900; the Green Dragon heals fully.
    wounds = initial.enemy_wounds.at[GUARD, 4].set(800).at[DRAGON, 1].set(500)
    rested, _ = step(initial.replace(enemy_wounds=wounds), jnp.int32(REST))
    assert rested.enemy_wounds[GUARD, 4] == 35 and rested.enemy_wounds[DRAGON, 1] == 0


def test_enemy_regeneration_uses_own_regen_and_place(default_game):
    """Original getUnitRegen (D2ModdingToolset unitutils.cpp): Gunits REGEN + place."""
    _, initial, step, _ = default_game
    nobody = initial.replace(territory_claims=jnp.zeros_like(initial.territory_claims))
    everywhere = initial.replace(territory_claims=jnp.full_like(initial.territory_claims, DEFAULT['size']**2))
    # (stack, slot, max HP): the Troll (REGEN 50), a Squire in a level 1 city, an Orc guarding a ruin.
    troll, city, ruin = (5, 1, 350), (21, 0, 100), (69, 0, 200)

    def healed(state, stack, slot, hp):
        hurt = state.replace(enemy_wounds=state.enemy_wounds.at[stack, slot].set(hp-1))
        rested, _ = step(hurt, jnp.int32(REST))
        return hp-1-int(rested.enemy_wounds[stack, slot])

    assert healed(nobody, *troll) == 210  # 50% + 10% on land nobody owns
    assert healed(everywhere, *troll) == 175  # no land bonus on the player's land
    for state in (nobody, everywhere):
        assert healed(state, *city) == 15  # 5% + 10% of a level 1 city, never the land bonus
        assert healed(state, *ruin) == 40  # GVars REGEN_RUIN: a fixed 20%


def test_capturing_both_objective_cities_wins(default_game):
    """Reference goal "cities" (maps/default.py FINAL_OBJECTIVE_CITIES)."""
    env, initial, step, _ = default_game
    names = [c['name'] for c in DEFAULT['cities']]
    goals = [names.index('Порт Полонис'), names.index('Соругирилла')]
    assert jnp.flatnonzero(env.territory.goal_cities).tolist() == goals
    # Other stacks, cities and the capital guard may remain.
    both = initial.city_owned.at[jnp.array(goals)].set(True)
    won, ts = step(initial.replace(city_owned=both), jnp.int32(REST))
    assert won.won and won.done and float(ts.reward) > 2
    one = initial.city_owned.at[goals[0]].set(True)
    cleared = initial.replace(alive=jnp.zeros_like(initial.alive),
                              city_owned=jnp.ones_like(one).at[goals[1]].set(False))
    open_, _ = step(cleared, jnp.int32(REST))
    assert not open_.won and not open_.done
    # Entering the last objective city after its defenders fell captures it and wins.
    city = DEFAULT['cities'][goals[1]]['position']
    blocked = env.territory.blocked_grid.reshape(50, 50)
    direction, (dr, dc) = next((i, d) for i, d in enumerate(DIRECTIONS)
                               if not blocked[city[0]-d[0], city[1]-d[1]])
    near = initial.replace(position=jnp.array([city[0]-dr, city[1]-dc], jnp.int32), city_owned=one,
                           alive=initial.alive & ~env.territory.city_enemies[goals[1]])
    entered, ts = step(near, jnp.int32(direction))
    assert entered.city_owned[goals[1]] and entered.won and entered.done and float(ts.reward) > 2


def test_captured_city_gives_its_vault_items(default_game):
    """Original .sg CMidVillage inventories go to the hero on capture (docs/DEFAULT_MAP.md)."""
    env, initial, step, _ = default_game
    names, items = [c['name'] for c in DEFAULT['cities']], env.item_rules
    assert [c['items'] for c in DEFAULT['cities']] == [  # Brigham's Port: the Legions' chalice event
        {}, {'unholy_chalice': 1}, {'staff_of_holiness': 1}, {'boots_speed': 1}, {'zombie_talisman': 1}]
    polonis, boots = names.index('Порт Полонис'), items.keys.index('boots_speed')
    city = DEFAULT['cities'][polonis]['position']
    blocked = env.territory.blocked_grid.reshape(50, 50)
    direction, (dr, dc) = next((i, d) for i, d in enumerate(DIRECTIONS)
                               if not blocked[city[0]-d[0], city[1]-d[1]])
    near = initial.replace(position=jnp.array([city[0]-dr, city[1]-dc], jnp.int32),
                           alive=initial.alive & ~env.territory.city_enemies[polonis])
    entered, _ = step(near, jnp.int32(direction))
    assert entered.city_owned[polonis] and not entered.done  # one of the two goal cities
    assert entered.last_item_loot[boots] == 1 and entered.item_inventory.tolist().count(boots) == 1
    # An owned city's vault is empty: entering it again adds nothing.
    again, _ = step(near.replace(city_owned=entered.city_owned, item_inventory=entered.item_inventory),
                    jnp.int32(direction))
    assert again.last_item_loot[boots] == 0 and again.item_inventory.tolist().count(boots) == 1


def test_without_objective_cities_the_capital_guard_is_not_a_goal():
    env = NumberGrid(map_config={**DEFAULT, 'cities': [dict(c, objective=False) for c in DEFAULT['cities']]})
    only_guard = jnp.zeros(len(DEFAULT['opponent_positions']), bool).at[GUARD].set(True)
    assert not env.territory.objective_cities
    spawned = ~env.scenario.scheduled  # Same initial spawn state as reset().
    assert not env._goals_left(only_guard, spawned)
    assert env._goals_left(only_guard.at[0].set(True), spawned)


def test_reference_hero_skills(default_game):
    """docs/HERO_SKILLS.md: Pathfinding, the warrior's Might at 9, Weapon Master at 12, equipment lore."""
    env, initial, step, _ = default_game
    progression, hero = env.progression, int(initial.unit_ids[1])
    row, plain = progression.rows[hero], progression.ids['duke']
    assert row['key'] == 'duke' and hero != plain  # the catalogue row keeps no Might
    assert [b['level'] for b in row['level_bonuses']] == [4, 5, 9, 13, 14, 15]
    assert [s['level'] for s in row['hero_skills']] == list(range(1, 16))
    assert row['hero_skills'][8]['key'] == 'might' and row['hero_skills'][0]['key'] == 'artifact_knowledge'
    damage = lambda i, level: float(progression.stat_levels[level, i, DAMAGE])
    assert damage(hero, 8) == damage(plain, 8)
    assert damage(hero, 9) == round(damage(plain, 9)*1.25)  # once, Python round
    assert damage(hero, 10)-damage(hero, 9) == damage(plain, 10)-damage(plain, 9)
    # A hero starting past the Might level gets it at once.
    started = ProgressionRules(dict(env.map_config, hero_combat_stats=[{}, dict(level=10), {}, {}]), env.construction)
    late = started.initial_ids[1]
    assert started.base_levels[late] == 10
    assert float(started.stat_levels[10, late, DAMAGE]) == round(progression.rows[plain]['damage']*1.25)
    # Pathfinding from level 2: +4 points, round(20% of 20), on top of +1 per level.
    levels = lambda level: initial.replace(unit_levels=initial.unit_levels.at[1].set(level))
    assert [int(env.movement_cap(levels(n))) for n in (1, 2, 10, 12)] == [20, 25, 33, 33]
    # Warrior: artifacts from the start, banner 7, book 11.
    rules = env.item_rules
    stocked = initial.replace(item_inventory=jnp.array(
        [rules.keys.index(k) for k in ('ring_of_ages', 'banner_war', 'tome_arcanum')]+[-1]*(rules.capacity-3), jnp.int32))
    choose = jax.jit(rules.choose)
    worn = lambda level: [rules.keys[i] if i >= 0 else None for i in choose(stocked, True, jnp.int32(level)).tolist()]
    assert worn(1) == ['ring_of_ages', None, None, None, None]
    assert worn(7) == ['ring_of_ages', None, 'banner_war', None, None]
    assert worn(11) == ['ring_of_ages', None, 'banner_war', 'tome_arcanum', None]
    # Weapon Master: +25% battle XP for the party from level 12, as Tome of War.
    battle, _ = step(initial.replace(position=jnp.array([28, 7], jnp.int32)), jnp.int32(1))
    assert battle.in_battle and battle.enemy == 11

    def shares(level):
        won, _ = step(battle.replace(actor=jnp.int32(1), unit_levels=battle.unit_levels.at[1].set(level),
                                     hp=battle.hp.at[6:].set(0), battle_xp=jnp.array([0, 120], jnp.int32),
                                     battle_earned_xp=jnp.where(battle.hp > 0,30,0)),
                      jnp.int32(DEFEND))
        for _ in range(6):
            if won.in_battle:
                won, _ = step(won, jnp.int32(WAIT))  # post-battle healing turns, then the award
        assert not won.in_battle
        return won.last_xp[:6].tolist()
    assert shares(11) == [30, 30, 30, 0, 30, 0]
    assert shares(12) == [37, 37, 37, 0, 37, 0]  # Original bonus truncates 30*25/100.
