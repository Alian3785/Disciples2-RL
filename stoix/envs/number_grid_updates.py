"""Exact replacements for small .at[].set updates and table lookups in the batched battle loop.

A scatter on a 12-slot array costs a GPU kernel launch plus a buffer copy, and a lookup with
several index arrays first concatenates them in a kernel of its own; the loop's cost per pass is
set by the number of such launches. These forms compute the same values but fuse with the
surrounding elementwise work.
"""
import jax.numpy as jnp


def take_rows(table, *indices):
    """table[i, j, ...] for (broadcastable) index arrays over its leading axes.

    Each index is normalised and clamped to its axis exactly like jnp indexing, then combined
    into one row number of the flattened table, so the lookup is a single-index gather.
    """
    flat = 0
    for size, index in zip(table.shape, indices):
        index = jnp.asarray(index)
        index = jnp.clip(jnp.where(index < 0, index+size, index), 0, size-1)
        flat = flat*size+index
    return table.reshape((-1,)+table.shape[len(indices):])[flat]


def set_column(values, column, new):
    """values with its last-axis entry `column` (static) replaced by `new`, cast like .at[].set.

    Equal to values.at[..., column].set(new), but an elementwise select: on the GPU it fuses
    with its neighbours instead of launching a scatter and a buffer copy per update.
    """
    selected = jnp.arange(values.shape[-1]) == column
    return jnp.where(selected, jnp.expand_dims(jnp.asarray(new).astype(values.dtype), -1), values)


def set_rows(values, rows, new):
    """values.at[rows].set(new) for one row index or a short vector of distinct ones.

    Negative rows count from the end and out-of-range rows are dropped, as with .at[].set,
    but each row is an elementwise select: on the GPU the update fuses with its neighbours
    instead of launching a scatter and a buffer copy.
    """
    rows = jnp.asarray(rows)
    size = values.shape[0]
    rows = jnp.where(rows < 0, rows+size, rows)
    index = jnp.arange(size).reshape((size,)+(1,)*(values.ndim-1))
    new = jnp.asarray(new).astype(values.dtype)
    if rows.ndim == 0:
        return jnp.where(index == rows, new, values)
    new = jnp.broadcast_to(new, rows.shape+values.shape[1:])
    if rows.shape[0] > 8 and values.ndim == 1 and not jnp.issubdtype(values.dtype, jnp.floating):
        # Long index vectors (inventory compaction): one comparison matrix instead of a chain;
        # each row matches at most one index, so the masked sum is that index's value.
        hits = index[:, None] == rows
        if values.dtype == jnp.bool_:
            return jnp.where(jnp.any(hits, axis=1), jnp.any(hits & new, axis=1), values)
        return jnp.where(jnp.any(hits, axis=1), jnp.sum(jnp.where(hits, new, 0), axis=1, dtype=values.dtype), values)
    for j in range(rows.shape[0]):
        values = jnp.where(index == rows[j], new[j], values)
    return values
