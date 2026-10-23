"""Shared wire schema for GPU events and the CPU report writer; no JAX imports."""

SCHEMA_VERSION = 1
HEADER = ('step', 'day', 'outcome', 'enemy', 'method', 'battle_result',
          'defeated_enemy', 'row', 'column', 'action')
# Outcomes: ongoing, victory, defeat, timeout. Methods: none, hero, summon, spell.
# Battle results: none, victory, defeat, retreat.
WIDTH = len(HEADER) + 24 + 30 + 6 + 6 + 6
