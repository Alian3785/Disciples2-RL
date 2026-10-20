"""Shared one-hit healer wards with caster ownership and sticky native use."""
import jax.numpy as jnp

# Matches combat source bits; keeping ownership separate prevents stacked blocks.
# Installed GAttacks WARD1..4 and GmodifL: healers bestow only fire, air, water
# and earth wards (Sundancer fire, Sylfid air, Deva roshi all four).
ELEMENT_BITS = (4, 256, 8, 2)  # fire, air, water, earth
WARD_MASK = sum(ELEMENT_BITS)  # 270


def ward_tags(owners):
    return _element_tags(owners != 0)


def _element_tags(active):
    return ((active[...,0].astype(jnp.uint32) << 2) | (active[...,1].astype(jnp.uint32) << 8)
            | (active[...,2].astype(jnp.uint32) << 3) | (active[...,3].astype(jnp.uint32) << 1))


def grant_wards(owners,native_used,used,native,recipients,elements,caster):
    bits = jnp.array(ELEMENT_BITS,jnp.uint32)
    selected = recipients[:,None] & ((elements & bits) != 0)
    tags = jnp.where(recipients,elements & jnp.uint32(WARD_MASK),jnp.uint32(0))
    caster_bit = jnp.left_shift(jnp.uint32(1),caster.astype(jnp.uint32))
    owners = owners | jnp.where(selected,caster_bit,jnp.uint32(0))
    native_used |= used & native & tags
    used &= ~tags
    return owners,native_used,used


def expire_wards(owners,native_used,used,native,casters):
    kept = owners & ~casters
    expired = _element_tags((owners != 0) & (kept == 0))
    # A native block consumed before a refresh must stay spent after expiry.
    used = (used & ~expired) | ((used | native_used) & native & expired)
    return kept,native_used & ~expired,used
