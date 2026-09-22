import pytest
from gymnasium import spaces
from campaign_env import CampaignEnv


class AliasEnv(CampaignEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_chests = dict(self._static_chests)
        self._static_chests[(1, 1)] = tuple(value for item_id, name, canonical in self.ORIGINAL_ITEM_BINDINGS for value in (item_id, name))
        self._refresh_dynamic_action_layout()
        self.action_space = spaces.Discrete(self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT)


@pytest.mark.parametrize('item_id,original,canonical', CampaignEnv.ORIGINAL_ITEM_BINDINGS)
def test_original_names_and_ids_share_inventory_count_value_and_consumption(item_id, original, canonical):
    env = CampaignEnv(log_enabled=False)
    env.reset(seed=123)
    env.heroitems.extend([original, {'item_id': item_id.lower(), 'name': 'Translated name'}])
    for name in (original, item_id, canonical):
        assert env._count_hero_item(name) == 2
        assert env._hero_item_gold_value_by_name(name) == env._hero_item_gold_value_by_name(canonical)
        assert not env._is_sell_only_hero_item(env._make_hero_item_entry(name))
    assert env._consume_hero_item(canonical)
    assert env._consume_hero_item(original)
    assert env._count_hero_item(item_id) == 0


def test_aliases_reserve_actions_and_apply_book_potion_and_staff_effects():
    env = AliasEnv(log_enabled=False, persist_blue_hp=True)
    env.reset(seed=123)
    hero = env._resolve_travel_hero()
    hero['hero_abilities'] = [env.HERO_BOOK_LORE_ABILITY_KEY, env.HERO_SORCERY_LORE_ABILITY_KEY]
    env._add_hero_item('Tome of Thought')
    assert env.equipped_book_items == [env.TOME_OF_MIND_ITEM_NAME]
    assert env.scenario_book_item_names.count(env.TOME_OF_MIND_ITEM_NAME) == 1
    env._add_hero_item('g000ig0003')
    action = env._potion_action_start_for_item('Treebark Potion') + env.GRID_POTION_USE_POSITIONS.index(7)
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info['potion_applied'] and info['potion_consumed']
    assert env._count_hero_item('Treebark Potion') == 0
    env._init_battle(enemy_id=1)
    unit = next(u for u in env.battle_env.combined if u['team'] == 'blue' and u['position'] == 7)
    assert unit['armor'] == env.blue_team_state[0]['armor'] + 30
    assert unit.get('incoming_damage_multiplier', 1.0) == 1.0
    battle_hero = next(u for u in env.battle_env.combined if u['team'] == 'blue' and u.get('hero'))
    assert 'Mind' in battle_hero['resistance']
    for original, spell in [('Staff of Fumbling', 'elf_d2_s015'), ('Staff of Treecalling', 'elf_d2_s007'), ('Staff of Tempest', 'mcl_d2_s019')]:
        entry = env._staff_spell_action_entry_for_item(original)
        assert entry['spell_id'] == spell
        assert sum(e['item_name'] == entry['item_name'] for e in env.staff_spell_action_entries()) == 2
