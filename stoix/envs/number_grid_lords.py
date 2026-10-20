"""Ruler bonuses and starting parties from the installed manual and Python reference."""
DEFAULT_FACTION = 'legions'
LORDS = {
    'warrior': dict(name='Воин', regeneration=15, research_divisor=1, max_spell_level=4,
        city_cost_divisor=1, starting_building=None,
        description='Регенерация отряда +15% в день. Изучение заклинаний I–IV уровней.'),
    'mage': dict(name='Маг', regeneration=0, research_divisor=2, max_spell_level=5,
        city_cost_divisor=1, starting_building='Башня магии',
        description='Изучение заклинаний I–V уровней за половину маны. Башня магии построена с начала игры.'),
    'guildmaster': dict(name='Глава гильдии', regeneration=0, research_divisor=1, max_spell_level=4,
        city_cost_divisor=2, starting_building='Гильдия',
        description='Улучшение городов за половину золота. Гильдия построена с начала игры. Изучение заклинаний I–IV уровней. Особые действия воров пока недоступны.'),
}
# Python reference campaign_env.py:_starting_blue_roster_spec, positions 7-12 as slots 0-5
# (front row 0-2, back row 3-5). The reference has no Legions warrior entry and falls back to
# grid.py BLUE_TEAM_TEMPLATE. Its optional boss roster (use_boss_starting_roster) is not ported.
STARTING_PARTIES = {
    'empire': dict(
        warrior=['squire', 'g000uu0019', 'squire', None, 'acolyte', None],
        mage=['squire', None, 'squire', 'g000uu0021', None, 'acolyte'],
        guildmaster=['squire', None, 'squire', 'g000uu0020', None, 'acolyte']),
    'legions': dict(
        warrior=['possessed', 'duke', 'possessed', None, 'cultist', None],
        mage=['possessed', None, 'possessed', 'g000uu0072', None, 'cultist'],
        guildmaster=['possessed', None, 'possessed', 'g000uu0071', None, 'cultist']),
    'mountain_clans': dict(
        warrior=['reference_g000uu0036', 'g000uu0044', 'reference_g000uu0036', None, 'travnitsa', None],
        mage=['reference_g000uu0036', None, 'reference_g000uu0036', 'g000uu0046', None, 'travnitsa'],
        guildmaster=['reference_g000uu0036', None, 'reference_g000uu0036', 'g000uu0045', None, 'travnitsa']),
    'undead_hordes': dict(
        warrior=['g000uu0086', 'g000uu0096', 'g000uu0086', None, 'reference_g000uu0080', None],
        mage=['g000uu0086', None, 'g000uu0086', 'g000uu0098', None, 'reference_g000uu0080'],
        guildmaster=['g000uu0086', None, 'g000uu0086', 'nosferatu', None, 'reference_g000uu0080']),
    'elves': dict(
        warrior=['reference_g000uu8014', 'g000uu8009', 'reference_g000uu8014', None, 'medium', None],
        mage=['reference_g000uu8014', None, 'reference_g000uu8014', 'g000uu8010', None, 'medium'],
        guildmaster=['reference_g000uu8014', None, 'reference_g000uu8014', 'g000uu8011', None, 'medium']),
}


def lord_settings(game_map):
    kind = game_map.get('lord_type', 'warrior')
    if not isinstance(kind, str) or kind not in LORDS:
        raise ValueError('Unknown lord_type')
    return dict(id=kind, **LORDS[kind])


def with_starting_party(game_map):
    """Give a named-unit map without hero_roster the reference party of its faction and ruler.

    An explicit hero_roster stays a fixed scenario party, like a reference map starting_roster.
    hero_units defaults to the number of units in the roster.
    """
    if not game_map.get('unit_progression'):
        return game_map
    roster = game_map.get('hero_roster')
    if roster is None:
        faction = game_map.get('faction', DEFAULT_FACTION)
        if faction not in STARTING_PARTIES:
            raise ValueError(f'Unknown faction: {faction}')
        roster = STARTING_PARTIES[faction][lord_settings(game_map)['id']]
    if not isinstance(roster, list):
        raise ValueError('hero_roster must be a list of six slots')
    count = sum(unit is not None for unit in roster)
    return {**game_map, 'hero_roster': list(roster), 'hero_units': game_map.get('hero_units', count)}
