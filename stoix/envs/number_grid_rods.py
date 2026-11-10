"""Rods on resource sources, planted by a hired Baroness (user rule for S143, 09.11.2026).

On «Дикие горы» a gold mine beside the capital and the mana sources lie beyond the Legions'
land. As a simplification of the original rod planter, `hire_rod_planter` hires the Baroness
once, for the configured gold, with no unit on the map and no position requirement. After that,
`plant_rod_<k>` plants a rod on source k for its own price, without walking there: an affordable
rod on a source that has none and is not on the agent's land. The optional `sources` list limits
the rod sites (on «Дикие горы» the five mana sources nearest to the capital by walking distance,
user rule 09.11.2026), and `guards` names, per site, the enemy army nearest to that source: its
rod is allowed only once that army is defeated. From then on the source pays its daily income
(50 gold or mana) as if the agent's land covered it; the land itself, movement and battles do
not change. Rods stay until the episode ends. Each rod pays the optional `reward` once (user
request 09.11.2026; 1.0 on «Дикие горы»), at most once per source.
"""
import jax.numpy as jnp

ROD_PLANTER_HIRED, ROD_PLANTED = 63, 64
KEYS = {'name', 'cost', 'rod_cost', 'reward', 'sources', 'guards'}


def _spec(game_map):
    spec = game_map.get('rod_planter')
    if spec is None:
        return None
    if not isinstance(spec, dict) or not {'name', 'cost', 'rod_cost'} <= set(spec) <= KEYS:
        raise ValueError('rod_planter takes name, cost, rod_cost and optional reward, sources and guards')
    if not isinstance(spec['name'], str) or any(type(spec[k]) is not int or spec[k] <= 0 for k in ('cost', 'rod_cost')):
        raise ValueError('rod_planter needs a name and positive integer prices')
    if type(spec.get('reward', 0.)) not in (int, float) or not 0 <= spec.get('reward', 0.) < 100:
        raise ValueError('rod_planter reward must be a small nonnegative number')
    mines, armies = len(game_map.get('mines', [])), len(game_map['opponent_positions'])
    sources = spec.get('sources', list(range(mines)))
    guards = spec.get('guards', [-1]*len(sources))
    if (not sources or len(set(sources)) != len(sources) or any(type(s) is not int or not 0 <= s < mines for s in sources)
            or len(guards) != len(sources) or any(type(g) is not int or not -1 <= g < armies for g in guards)):
        raise ValueError('rod_planter sources are distinct mine indices, guards one army index (or -1) each')
    return dict(spec, sources=sources, guards=guards)


def rod_action_names(game_map):
    spec = _spec(game_map)
    if spec is None:
        return ()
    return ('hire_rod_planter',) + tuple(f'plant_rod_{k}' for k in spec['sources'])


class RodRules:
    def __init__(self, game_map, territory, start):
        spec = _spec(game_map)
        self.enabled = spec is not None
        self.start = self.end = start
        self.count = self.observation_size = 0
        if not self.enabled:
            return
        if territory is None or not territory.mines:
            raise ValueError('rod_planter requires territory with resource sources')
        self.territory = territory
        self.sources = jnp.array(spec['sources'], jnp.int32)
        self.guards = jnp.array(spec['guards'], jnp.int32)
        self.count = len(spec['sources'])
        self.cost, self.rod_cost = spec['cost'], spec['rod_cost']
        self.reward = float(spec.get('reward', 0.))
        self.plant_start = start+1
        self.end = self.plant_start+self.count
        self.observation_size = 1+2*self.count
        self.metadata = dict(start=start, plant_start=self.plant_start, name=spec['name'], cost=self.cost,
                             rod_cost=self.rod_cost, reward=self.reward, sources=spec['sources'], guards=spec['guards'])

    def guard_defeated(self, state):
        return (self.guards < 0) | ~state.alive[jnp.maximum(self.guards, 0)]

    def available(self, state):
        ready = ~state.in_battle & ~state.done
        hire = ready & ~state.rod_planter & (state.gold >= self.cost)
        land = self.territory.owned(state, self.territory.mine_positions)[self.sources]
        plant = (ready & state.rod_planter & ~state.mine_rods[self.sources] & ~land & self.guard_defeated(state)
                 & (state.gold >= self.rod_cost))
        return jnp.concatenate((hire[None], plant))

    def apply(self, state, action):
        """Returns the state and the planting reward (reference profiles pay it in their world reward)."""
        index = jnp.clip(action-self.start, 0, self.count)
        valid = (action >= self.start) & (action < self.end) & self.available(state)[index]
        hire, plant = valid & (index == 0), valid & (index > 0)
        source = self.sources[jnp.maximum(index-1, 0)]
        cost = jnp.where(hire, self.cost, jnp.where(plant, self.rod_cost, 0))
        return state.replace(rod_planter=state.rod_planter | hire,
            mine_rods=state.mine_rods | (plant & (jnp.arange(state.mine_rods.size) == source)),
            gold=state.gold-cost, last_service_cost=jnp.where(valid, cost, state.last_service_cost),
            last_target=jnp.where(plant, source, state.last_target),
            last_event=jnp.where(hire, ROD_PLANTER_HIRED, jnp.where(plant, ROD_PLANTED, state.last_event))
        ), jnp.where(plant, self.reward, 0.).astype(jnp.float32)

    def observation(self, state):
        return jnp.concatenate((state.rod_planter[None], state.mine_rods[self.sources],
                                self.guard_defeated(state))).astype(jnp.float32)
