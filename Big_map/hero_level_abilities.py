"""Fixed late-game hero skills; effects match GmodifL, levels are training rules."""

# level, key, display name, stat, multiplier, flat bonus
HERO_LEVEL_STAT_ABILITIES = (
    (13, 'first_strike', 'Первый удар (+50% инициативы)', 'initiative', 1.5, 0),
    (14, 'accuracy', 'Точность (+20% к точности)', 'accuracy', 1.2, 0),
    (15, 'natural_armor', 'Природная броня (+20 брони)', 'armor', 1.0, 20),
)


def apply_hero_level_stat_abilities(hero):
    """Grant each bonus once, including to heroes starting above its level.

    Update pre-buff snapshots too, so battle cleanup cannot erase a learned skill.
    The caller must check that this unit is a hero.
    """
    level = int(hero.get('Level', 0) or 0)
    learned = set(hero.get('hero_level_stat_abilities') or ())
    for required, key, _, stat, multiplier, bonus in HERO_LEVEL_STAT_ABILITIES:
        if level < required or key in learned:
            continue
        stats = (stat, 'accuracy_secondary') if stat == 'accuracy' else (stat,)
        for target in stats:
            if target == 'accuracy_secondary' and target not in hero:
                continue
            keys = {target}
            keys.update(k for k in hero if k.startswith('campaign_') and k.endswith('_base_' + target))
            if target == 'initiative':
                hero.setdefault('initiative_base', hero.get('initiative', 0))
                keys.add('initiative_base')
            elif target == 'armor':
                hero.setdefault('base_armor', hero.get('armor', 0))
                keys.update(k for k in ('base_armor', 'shatter_original_armor') if k in hero)
            previous = hero.get(target, 0)
            for field in keys:
                value = max(0, int(float(hero.get(field, previous) or 0) * multiplier + bonus + 0.5))
                hero[field] = min(100, value) if target.startswith('accuracy') else value
        learned.add(key)
    if learned:
        hero['hero_level_stat_abilities'] = sorted(learned)
