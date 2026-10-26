"""Equipment on-hit effects, sharing temporary forms with native unit attacks."""
import jax.numpy as jnp
from stoix.envs.number_grid_limits import MAX_HP
from stoix.envs.number_grid_combat import HP, INITIATIVE
from stoix.envs.number_grid_effects import source_protection, disable_attack_accuracy
from stoix.envs.number_grid_updates import set_rows


def transform_imp(env,state,used,slots,selected,long=True):
    """Apply an imp form without moving an attack already deferred by Wait."""
    first=selected & ~state.imp[slots] & (state.decay_form[slots]==0)
    mask=set_rows(jnp.zeros(12,bool),slots,selected)
    first_mask=set_rows(jnp.zeros(12,bool),slots,first)
    state=env._capture_initiative_form(state,mask,first_mask)
    state=env._capture_ward_forms(state,mask,first_mask)
    state=env._capture_damage_forms(state,mask,first_mask)
    saved=jnp.where(first,used[slots],state.imp_wards[slots])
    sizes=(env.progression.sizes[env._effective_ids(state)] if env.progression_enabled
           else env.size_table[state.enemy])
    priority=jnp.where(selected & (state.turn_phase[slots]==0),
                       jnp.where(sizes[slots]==2,50.,30.),state.priority[slots])
    state=_suspend_weakening(state,slots,selected,first)
    state=env._record_effect(state,slots,selected,4)
    state=state.replace(transform_long=set_rows(state.transform_long,slots,jnp.where(selected,long,state.transform_long[slots])),
        imp=set_rows(state.imp,slots,state.imp[slots]|selected),
        imp_wards=set_rows(state.imp_wards,slots,saved),
        priority=set_rows(state.priority,slots,priority))
    return state,set_rows(used,slots,jnp.where(selected,jnp.uint32(0),used[slots]))


def transform_decay(env,state,hp,used,slots,selected):
    lower=env.progression.lower_forms[env._effective_ids(state)[slots]]
    lower=jnp.where(state.fenrir[slots] & (state.decay_form[slots]==0),0,lower)
    return transform_form(env,state,hp,used,slots,selected,lower)


def transform_form(env,state,hp,used,slots,selected,forms):
    """Wight level-down and Orb of Lycanthropy: a temporary catalogue form."""
    selected &= forms>0
    first=selected & ~state.imp[slots] & (state.decay_form[slots]==0)
    old_max=env._combat_stats(state)[slots,HP]
    mask=set_rows(jnp.zeros(12,bool),slots,selected)
    first_mask=set_rows(jnp.zeros(12,bool),slots,first)
    state=env._capture_initiative_form(state,mask,first_mask)
    state=env._capture_ward_forms(state,mask,first_mask)
    state=env._capture_damage_forms(state,mask,first_mask)
    native=env.progression.base_stats[forms]
    scaled=jnp.rint(hp[slots]/jnp.maximum(old_max,1)*native[:,HP]).astype(jnp.int32)
    scaled=jnp.where(hp[slots]>0,jnp.clip(scaled,1,MAX_HP),0)
    hp=set_rows(hp,slots,jnp.where(selected,scaled,hp[slots]))
    saved=jnp.where(first,used[slots],state.imp_wards[slots])
    state=_suspend_weakening(state,slots,selected,first)
    state=state.replace(
        decay_form=set_rows(state.decay_form,slots,jnp.where(selected,forms,state.decay_form[slots])),
        decay_shreds=set_rows(state.decay_shreds,slots,jnp.where(selected,0,state.decay_shreds[slots])),
        imp=set_rows(state.imp,slots,state.imp[slots]&~selected),
        imp_wards=set_rows(state.imp_wards,slots,saved),
        priority=set_rows(state.priority,slots,jnp.where(selected & (state.turn_phase[slots]==0),native[:,INITIATIVE],state.priority[slots])))
    return state,hp,set_rows(used,slots,jnp.where(selected,jnp.uint32(0),used[slots])),selected


def _suspend_weakening(state,slots,selected,first):
    """A temporary form starts unweakened; the native form keeps its debuff."""
    update={}
    for name in ('weakened','item_weakened'):
        current,saved=getattr(state,name),getattr(state,'saved_'+name)
        update['saved_'+name]=set_rows(saved,slots,jnp.where(first,current[slots],saved[slots]))
        update[name]=set_rows(current,slots,current[slots]&~selected)
    return state.replace(**update)


def equipment_attack(env,state,hp,used,target,connected,inflicted,random_values):
    """Reference artifacts apply to successful single-target strikes only.

    Duplicate numeric artifacts stack, duplicate named combat artifacts do not.
    Random samples are independent of native hit/status/duration samples.
    """
    rules=env.item_rules
    actor=state.actor
    hero=env._equipment_hero(state)
    connected &= actor==hero
    if rules.has_drain:
        drain=connected & jnp.any(rules.value(state,'drain')[:2])
        hp=set_rows(hp,actor,jnp.where(drain,jnp.minimum(env.max_hp(state)[actor],hp[actor]+inflicted//4),hp[actor]))
    immune,warded=jnp.uint32(0),jnp.uint32(0)
    if not rules.status_items:
        return state,hp,used,immune,warded
    kinds={'paralysis':1,'poison':2,'imp':3,'decay':4}
    rows=({},)+rules.items
    kind_table=jnp.array([kinds.get(p.get('status'),0) for p in rows],jnp.int32)
    sources=jnp.array([p.get('source',0) for p in rows],jnp.uint32)
    chances=jnp.array([p.get('chance',0.) for p in rows])
    damages=jnp.array([p.get('poison',0) for p in rows],jnp.int32)
    supported={p['status'] for _,p in rules.status_items}
    slots=target[None]
    for i in range(2):
        item=state.equipped[i]
        kind=kind_table[item+1]
        chance=disable_attack_accuracy(chances[item+1], kind==1, state.round)
        hit=connected & (hp[target]>0) & (item>=0) & (random_values[80+2*i]<chance)
        if i:
            hit &= item!=state.equipped[0]
        # Soul Crystal uses the finite-paralysis helper; the Incubus horn uses
        # Abyss Devil's helper (which checks long paralysis before petrifying).
        hit &= (kind!=1) | jnp.where(sources[item+1]==7,~state.paralyzed[target],~state.long_paralyzed[target])
        hit &= (kind!=3) | ~env.progression.capital_guards[state.unit_ids[target]]
        hit &= (kind!=4) | ~env.progression.neutrals[state.unit_ids[target]]
        selected,used,blocked,protected=source_protection(hit[None],sources[item+1],env._combat_traits(state),used,slots)
        immune |= blocked
        warded |= protected
        if 'paralysis' in supported:
            state=env._record_disable(state,slots,selected & (kind==1))
            state=state.replace(paralyzed=set_rows(state.paralyzed,target,state.paralyzed[target] | (selected[0] & (kind==1))))
        if 'poison' in supported:
            poisoned=selected[0] & (kind==2)
            duration=2
            state=env._record_effect(state,slots,poisoned[None],0)
            state=state.replace(
                poison_mode=set_rows(state.poison_mode,target,jnp.where(poisoned,duration,state.poison_mode[target])),
                poison_damage=set_rows(state.poison_damage,target,jnp.where(poisoned,damages[item+1],state.poison_damage[target])),
                poison_source=set_rows(state.poison_source,target,jnp.where(poisoned,actor,state.poison_source[target])))
        if 'imp' in supported:
            state,used=transform_imp(env,state,used,slots,selected & (kind==3))
        if 'decay' in supported:
            state,hp,used,_=transform_decay(env,state,hp,used,slots,selected & (kind==4))
    return state,hp,used,immune,warded
