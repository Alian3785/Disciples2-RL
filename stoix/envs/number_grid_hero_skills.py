"""Reference hero skill ladder: one fixed skill per level, chosen by the lord's class.

Python reference (Big_map): campaign_env_data.py:178-205 levels and values,
campaign_env_economy.py:455-623 the _hero_has_* checks and the lord's Might.
The map flag ``hero_skills`` adds Pathfinding, Weapon Master and Might; the other
rungs already exist (leadership, level bonuses, equipment lore, sorcery lore).
"""
from stoix.envs.number_grid_lords import lord_settings

PATHFINDING_LEVEL, WEAPON_MASTER_LEVEL = 2, 12
PATHFINDING_PERCENT = 20  # of the hero's base movement, Python round
MIGHT_MULTIPLIER = 1.25
# Equipment slots (artifact, artifact, banner, book, boots) -> lore.
EQUIPMENT_LORE = ('artifact_knowledge', 'artifact_knowledge', 'banner_bearer', 'book_lore', 'marching_lore')
# A lord class knows one lore from the start and takes Might at that lore's level.
LORD_LORE = {'warrior': 'artifact_knowledge', 'mage': 'sorcery_lore', 'guildmaster': 'marching_lore'}
LADDER = (
    (2, 'pathfinding', 'Нахождение пути: +20% движения'),
    (3, 'leadership', 'Лидерство: +1 место в отряде'),
    (4, 'endurance', 'Выносливость: +20% HP'),
    (5, 'strength', 'Сила: +25% урона'),
    (6, 'leadership', 'Лидерство: ещё +1 место в отряде'),
    (7, 'banner_bearer', 'Знаменосец: знамёна'),
    (8, 'marching_lore', 'Походные знания: сапоги'),
    (9, 'artifact_knowledge', 'Знание артефактов: артефакты'),
    (10, 'sorcery_lore', 'Колдовские знания: сферы и талисманы'),
    (11, 'book_lore', 'Колдовство: книги'),
    (12, 'weapon_master', 'Мастер оружия: +25% опыта отряду'),
    (13, 'first_strike', 'Первый удар: +50% инициативы'),
    (14, 'accuracy', 'Точность: +20% к точности'),
    (15, 'natural_armor', 'Природная броня: +20 брони'),
)
LEVELS = {key: level for level, key, _ in LADDER}
MIGHT_NAME = 'Мощь: +25% урона'


def enabled(game_map):
    return bool(game_map.get('hero_skills', False))


def lore_levels(game_map):
    """Hero level that unlocks each equipment slot; 1 for the lord's own lore."""
    lore = LORD_LORE[lord_settings(game_map)['id']]
    starting = set(game_map.get('starting_hero_abilities', [])) | {lore}
    if starting - set(LEVELS):
        raise ValueError('Unknown starting hero ability')
    return tuple(1 if key in starting else LEVELS[key] for key in EQUIPMENT_LORE)


def might_bonus(game_map):
    """Might: hero damage x1.25 once, at the level of the lore the lord already knows.

    Reference _apply_lord_replacement_might_bonus: int(round(damage*1.25)).
    """
    level = LEVELS[LORD_LORE[lord_settings(game_map)['id']]]
    return dict(level=level, name=MIGHT_NAME, stat='damage', multiplier=MIGHT_MULTIPLIER,
                bonus=0, rounding='even')


def skill_list(game_map):
    """Display order: the lord's lore at level 1, then the ladder with Might in its place."""
    lore = LORD_LORE[lord_settings(game_map)['id']]
    names = dict((key, name) for _, key, name in LADDER)
    skills = [dict(level=1, key=lore, name=names[lore]+' (класс лорда)')]
    skills += [dict(level=1, key=key, name=names[key]+' (сценарий)')
               for key in game_map.get('starting_hero_abilities', []) if key != lore]
    for level, key, name in LADDER:
        skills.append(dict(level=level, key='might', name=MIGHT_NAME) if key == lore
                      else dict(level=level, key=key, name=name))
    return skills
