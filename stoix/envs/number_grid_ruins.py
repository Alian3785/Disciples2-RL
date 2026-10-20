"""Guarded 3x3 ruins with one-shot loot. Sources: docs/RUINS.md."""
import jax.numpy as jnp
from stoix.envs.number_grid_limits import add_resource

from stoix.envs.number_grid_sites import APPROACH_OFFSETS


def ruin_geometry(game_map):
    ruins = game_map.get('ruins', [])
    if not isinstance(ruins, list):
        raise ValueError('ruins must be a list')
    result = []
    for ruin in ruins:
        anchor = ruin.get('position') if isinstance(ruin, dict) else None
        if (not isinstance(anchor, (list, tuple)) or len(anchor) != 2
                or any(type(n) is not int or not 0 < n < game_map['size']-4 for n in anchor)):
            raise ValueError('Ruins need an interior 3x3 position and space for the entrance')
        r,c = anchor
        entrance = ruin.get('entrance', [r+2,c+2])
        if entrance not in ([r+2,c], [r+2,c+1], [r+2,c+2]):
            raise ValueError('Ruin entrance must be on its south edge')
        approaches = [(r+i,c+j) for i,j in APPROACH_OFFSETS] if entrance == [r+2,c+2] else [
            (r+3, entrance[1]+dc) for dc in (-1,0,1)]
        result.append(dict(ruin, name=ruin.get('name','Руины'), size=[3,3],
            entrance=entrance, footprint=[(r+i,c+j) for i in range(3) for j in range(3)],
            interaction_tiles=approaches))
    return result


class RuinRules:
    def __init__(self, game_map, potions, items, progression, sites=None):
        self.ruins = ruin_geometry(game_map)
        self.count = len(self.ruins)
        occupied = {tuple(game_map['agent_position'])}
        occupied.add(tuple(game_map.get('capital_position', game_map['agent_position'])))
        occupied.update(tuple(c['position']) for c in game_map.get('chests',[])+game_map.get('cities',[])+game_map.get('mines',[]))
        occupied.update(tuple(p) for p in game_map.get('obstacles',[]))
        if sites is not None:
            occupied.update(tuple(p) for site in sites.sites for p in site['footprint'])
        enemies = game_map['opponent_positions']
        indices = []
        for ruin in self.ruins:
            cells = set(ruin['footprint'])
            at = [i for i,p in enumerate(enemies) if tuple(p) in cells]
            if len(at) != 1 or list(enemies[at[0]]) != ruin['entrance']:
                raise ValueError('Each ruin requires exactly one guardian army at its lower-right entrance')
            if cells & occupied:
                raise ValueError('Ruin footprints must not overlap another object')
            occupied.update(cells)
            gold = ruin.get('gold',0)
            if type(gold) is not int or not 0 <= gold <= 1000000:
                raise ValueError('Ruin gold must be an integer between 0 and 1000000')
            ruin['gold'], ruin['enemy'] = gold, at[0]
            indices.append(at[0])
        blocked = {tuple(p) for p in game_map.get('obstacles',[])}
        blocked.update(p for ruin in self.ruins for p in ruin['footprint'])
        if sites is not None:
            blocked.update(p for site in sites.sites for p in site['footprint'])
        for ruin in self.ruins:
            ruin['interaction_tiles'] = [p for p in ruin['interaction_tiles'] if p not in blocked]
            if not ruin['interaction_tiles']:
                raise ValueError('Ruin entrance is blocked')
        self.enemies = jnp.array(indices,jnp.int32)
        self.enemy_ruin = jnp.array([indices.index(i) if i in indices else -1 for i in range(len(enemies))],jnp.int32)
        self.footprints = jnp.array([p for ruin in self.ruins for p in ruin['footprint']],jnp.int32)
        self.entrances = jnp.array([r['entrance'] for r in self.ruins],jnp.int32)
        self.approaches = jnp.array([r['interaction_tiles']+[(-1,-1)]*(5-len(r['interaction_tiles'])) for r in self.ruins],jnp.int32)
        self.heroes = jnp.array([bool(r.get('hero')) for r in progression.rows])
        self.loot = jnp.array([[r.get('potions',{}).get(k,0) for k in potions.keys] for r in self.ruins],jnp.int32)
        self.gold = jnp.array([r['gold'] for r in self.ruins],jnp.int32)
        self.items = items
        item_loot = items.ruin_loot[:,items.loot_kinds] if items is not None else jnp.zeros((self.count,0),jnp.int32)
        self.features = jnp.concatenate((jnp.array([r['position'] for r in self.ruins])/(game_map['size']-1),
            self.loot.astype(jnp.float32),item_loot.astype(jnp.float32),self.gold[:,None]/1000.),axis=1)
        self.observation_size = (self.features.shape[1]+1)*self.count
        self.metadata = dict(ruins=self.ruins, clear_reward=.25)

    def blocked(self, positions):
        return jnp.any(jnp.all(positions[...,None,:] == self.footprints,axis=-1),axis=-1)

    def attackable(self, state, positions):
        leader = jnp.any(self.heroes[state.unit_ids[:6]] & (state.hp[:6]>0)
                         & (state.summon_owner[:6]<0) & ~state.copied[:6])
        near = jnp.any(jnp.all(state.position == self.approaches,axis=-1),axis=-1)
        valid = near & leader & state.alive[self.enemies] & ~state.ruin_looted
        return jnp.any(jnp.all(positions[...,None,:] == self.entrances,axis=-1) & valid,axis=-1)

    def grant(self, state, victory):
        collected = victory & (self.enemies == state.enemy) & ~state.ruin_looted
        loot = jnp.sum(jnp.where(collected[:,None],self.loot,0),axis=0)
        if self.items is not None:
            state = self.items.collect_ruins(state,collected)
        return state.replace(ruin_looted=state.ruin_looted|collected,
            last_ruin=jnp.where(jnp.any(collected),jnp.argmax(collected),-1).astype(jnp.int32),
            potions=state.potions+loot,last_loot=loot,
            gold=add_resource(state.gold,jnp.sum(jnp.where(collected,self.gold,0)))), .25*jnp.any(collected)

    def observation(self, state):
        return jnp.concatenate((self.features,state.ruin_looted[:,None]),axis=1).reshape(-1)
