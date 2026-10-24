"""Scrolls and staves: hero items that cast a catalogue spell. See docs/SCROLLS_AND_STAVES.md.

Data: installed GItem.dbf (scripts/import_spell_items.py). Only items whose
spell can be researched and cast (stoix/envs/data/spells.json) are included:
101 scrolls and 14 staves. Rules: Python reference campaign_env_magic.py
_can_cast_scroll_spell/_can_cast_staff_spell, _step_unlock_scroll_magic.
"""
import json
from pathlib import Path

from stoix.envs.number_grid_spells import SPELL_BY_ID
from stoix.envs.number_grid_territory import RESOURCE_NAMES

DATA = json.loads((Path(__file__).parent / 'data/spell_items.json').read_text(encoding='utf-8'))
# campaign_env_data.py SCROLL_MAGE_UNLOCK_GOLD_COST: a non-mage hero hires a scroll reader once.
SCROLL_MAGE_COST = 500
SCROLL_MAGE_HIRED = 54
LORE_LEVEL = 10  # HERO_SORCERY_LORE_LEVEL: staves, like orbs and talismans
ROMAN = ('', 'I', 'II', 'III', 'IV', 'V')


def _label(item):
    spell = SPELL_BY_ID[item['spell']]
    name = f'«{spell["name"]}», {ROMAN[spell["level"]]}'
    if item['category'] == 'scroll':
        return f'Свиток {name}: без маны, расходуется'
    mana = ', '.join(f'{RESOURCE_NAMES[1+i].removeprefix("Мана ")} {n}' for i, n in enumerate(item['cast_mana']) if n)
    return f'Посох {name}: мана {mana} за применение'


SPELL_ITEMS = tuple(dict(item, label=_label(item)) for item in DATA['items'])
SPELL_ITEM_BY_KEY = {item['key']: item for item in SPELL_ITEMS}


def scenario_spell_items(game_map):
    """Catalogue order: scrolls, then staves, among the scenario's obtainable items."""
    from stoix.envs.number_grid_potions import merchants
    sources = [game_map.get('initial_items', {})] + [c.get('items', {}) for c in game_map.get('chests', [])
                                                       + game_map.get('ruins', []) + game_map.get('cities', [])
                                                       + merchants(game_map) if isinstance(c, dict)]
    keys = {key for contents in sources for key, count in contents.items() if count}
    return tuple(item for item in SPELL_ITEMS if item['key'] in keys)


def spell_item_action_names(items):
    if not items:
        return ()
    return ('hire_scroll_mage',) + tuple('cast_' + item['key'] for item in items)
