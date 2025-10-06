#Тиамат против дьявола бездны

UNITS_RED = [
    {"name": "Мертвый дракон", "initiative": 35, "initiative_base": 35, "team": "red", "position": 1, "stand": "ahead",
     "unit_type": "Dead dragon", "damage": 70, "damage_secondary": 25, "health": 0, "max_health": 0, "armor": 10,
     "accuracy": 80, "accuracy_secondary": 40, "immunity": ["death"], "resistance": [], "attack_type_primary": "death",
     "attack_type_secondary": "poison", "big": True, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Тиамат", "initiative": 55, "initiative_base": 55, "team": "red", "position": 2, "stand": "behind",
     "unit_type": "Tiamat", "damage": 50, "damage_secondary": 0, "health": 400, "max_health": 400, "armor": 0,
     "accuracy": 90, "accuracy_secondary": 90, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "Mind", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "воин", "initiative": 48, "initiative_base": 48, "team": "red", "position": 3, "stand": "behind",
     "unit_type": "Warrior", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 78, "accuracy_secondary": 0, "immunity": ["death"], "resistance": ["Mind"], "attack_type_primary": "Mind",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Смерть", "initiative": 40, "initiative_base": 40, "team": "red", "position": 4, "stand": "behind",
     "unit_type": "Death", "damage": 95, "damage_secondary": 20, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 55, "immunity": ["Weapon", "death"], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "poison", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 5, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 6, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]

UNITS_BLUE = [
    {"name": "Дьявол бездны", "initiative": 95, "initiative_base": 95, "team": "blue", "position": 7, "stand": "ahead",
     "unit_type": "Abyss Devil", "damage": 100, "damage_secondary": 0, "health": 330, "max_health": 330, "armor": 0,
     "accuracy": 50, "accuracy_secondary": 90, "immunity": [], "resistance": ["Mind"], "attack_type_primary": "Weapon",
     "attack_type_secondary": "Mind", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Боевой маг", "initiative": 62, "initiative_base": 62, "team": "blue", "position": 8, "stand": "behind",
     "unit_type": "Mage", "damage": 60, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 88, "accuracy_secondary": 0, "immunity": [], "resistance": ["Mind"],
     "attack_type_primary": "Fire", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Инквизитор", "initiative": 52, "initiative_base": 52, "team": "blue", "position": 9, "stand": "behind",
     "unit_type": "Uter", "damage": 70, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 5,
     "accuracy": 82, "accuracy_secondary": 65, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "Mind", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Арбалетчик", "initiative": 40, "initiative_base": 40, "team": "blue", "position": 10, "stand": "behind",
     "unit_type": "Archer", "damage": 75, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 90, "accuracy_secondary": 0, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "blue", "position": 11, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "blue", "position": 12, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]

#Солнечная танцовщица и два противника

UNITS_RED = [
    {"name": "Мертвый дракон", "initiative": 35, "initiative_base": 35, "team": "red", "position": 1, "stand": "ahead",
     "unit_type": "Dead dragon", "damage": 70, "damage_secondary": 25, "health": 0, "max_health": 0, "armor": 10,
     "accuracy": 80, "accuracy_secondary": 40, "immunity": ["death"], "resistance": [], "attack_type_primary": "death",
     "attack_type_secondary": "poison", "big": True, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "воин1", "initiative": 55, "initiative_base": 55, "team": "red", "position": 2, "stand": "behind",
     "unit_type": "Warrior", "damage": 70, "damage_secondary": 0, "health": 320, "max_health": 320, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Fire",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "воин", "initiative": 48, "initiative_base": 48, "team": "red", "position": 3, "stand": "behind",
     "unit_type": "Warrior", "damage": 50, "damage_secondary": 0, "health": 300, "max_health": 300, "armor": 0,
     "accuracy": 78, "accuracy_secondary": 0, "immunity": ["death"], "resistance": ["Mind"], "attack_type_primary": "Fire",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "арбалетчик", "initiative": 40, "initiative_base": 40, "team": "red", "position": 4, "stand": "behind",
     "unit_type": "Archer", "damage": 95, "damage_secondary": 20, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 55, "immunity": ["Weapon", "death"], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "poison", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 5, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 6, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]

UNITS_BLUE = [
    {"name": "Инквизитор", "initiative": 95, "initiative_base": 95, "team": "blue", "position": 7, "stand": "ahead",
     "unit_type": "Warrior", "damage": 100, "damage_secondary": 0, "health": 200, "max_health": 200, "armor": 0,
     "accuracy": 50, "accuracy_secondary": 90, "immunity": [], "resistance": ["Mind"], "attack_type_primary": "Weapon",
     "attack_type_secondary": "Mind", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Боевой маг", "initiative": 62, "initiative_base": 62, "team": "blue", "position": 8, "stand": "ahead",
     "unit_type": "Warrior", "damage": 50, "damage_secondary": 0, "health": 200, "max_health": 200, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 0, "immunity": [], "resistance": ["Weapon"],
     "attack_type_primary": "Fire", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Инквизитор", "initiative": 52, "initiative_base": 52, "team": "blue", "position": 9, "stand": "ahead",
     "unit_type": "Warrior", "damage": 70, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 5,
     "accuracy": 82, "accuracy_secondary": 65, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "Mind", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Арбалетчик", "initiative": 40, "initiative_base": 40, "team": "blue", "position": 10, "stand": "behind",
     "unit_type": "Archer", "damage": 75, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 90, "accuracy_secondary": 0, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Солнечная танцовщица", "initiative": 20, "initiative_base": 20, "team": "blue", "position": 11, "stand": "behind",
     "unit_type": "Sundancer", "damage": 40, "damage_secondary": 0, "health": 500, "max_health": 500, "armor": 0,
     "accuracy": 100, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Life",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "blue", "position": 12, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]


#Дева рощи и пример truncated


UNITS_RED = [
    {"name": "Мертвый дракон", "initiative": 35, "initiative_base": 35, "team": "red", "position": 1, "stand": "ahead",
     "unit_type": "Dead dragon", "damage": 70, "damage_secondary": 25, "health": 0, "max_health": 0, "armor": 10,
     "accuracy": 80, "accuracy_secondary": 40, "immunity": ["death"], "resistance": [], "attack_type_primary": "death",
     "attack_type_secondary": "poison", "big": True, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "воин1", "initiative": 55, "initiative_base": 55, "team": "red", "position": 2, "stand": "behind",
     "unit_type": "Warrior", "damage": 70, "damage_secondary": 0, "health": 320, "max_health": 320, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Fire",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "воин", "initiative": 48, "initiative_base": 48, "team": "red", "position": 3, "stand": "behind",
     "unit_type": "Warrior", "damage": 50, "damage_secondary": 0, "health": 300, "max_health": 300, "armor": 0,
     "accuracy": 78, "accuracy_secondary": 0, "immunity": ["death"], "resistance": ["Mind"], "attack_type_primary": "Fire",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "арбалетчик", "initiative": 40, "initiative_base": 40, "team": "red", "position": 4, "stand": "behind",
     "unit_type": "Archer", "damage": 95, "damage_secondary": 20, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 55, "immunity": ["Weapon", "death"], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "poison", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 5, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "red", "position": 6, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]

UNITS_BLUE = [
    {"name": "Инквизитор", "initiative": 95, "initiative_base": 95, "team": "blue", "position": 7, "stand": "ahead",
     "unit_type": "Warrior", "damage": 100, "damage_secondary": 0, "health": 200, "max_health": 200, "armor": 0,
     "accuracy": 50, "accuracy_secondary": 90, "immunity": [], "resistance": ["Mind"], "attack_type_primary": "Weapon",
     "attack_type_secondary": "Mind", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Боевой маг", "initiative": 62, "initiative_base": 62, "team": "blue", "position": 8, "stand": "ahead",
     "unit_type": "Warrior", "damage": 50, "damage_secondary": 0, "health": 200, "max_health": 200, "armor": 0,
     "accuracy": 80, "accuracy_secondary": 0, "immunity": [], "resistance": ["Weapon"],
     "attack_type_primary": "Fire", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Инквизитор", "initiative": 52, "initiative_base": 52, "team": "blue", "position": 9, "stand": "ahead",
     "unit_type": "Warrior", "damage": 70, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 5,
     "accuracy": 82, "accuracy_secondary": 65, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "Mind", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Арбалетчик", "initiative": 40, "initiative_base": 40, "team": "blue", "position": 10, "stand": "behind",
     "unit_type": "Archer", "damage": 75, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 90, "accuracy_secondary": 0, "immunity": [], "resistance": [],
     "attack_type_primary": "Weapon", "attack_type_secondary": "", "big": False,
     "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "Дева рощи", "initiative": 20, "initiative_base": 20, "team": "blue", "position": 11, "stand": "behind",
     "unit_type": "Deva roshi", "damage": 40, "damage_secondary": 0, "health": 500, "max_health": 500, "armor": 0,
     "accuracy": 100, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Life",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},

    {"name": "пусто", "initiative": 0, "initiative_base": 0, "team": "blue", "position": 12, "stand": "behind",
     "unit_type": "Archer", "damage": 0, "damage_secondary": 0, "health": 0, "max_health": 0, "armor": 0,
     "accuracy": 0, "accuracy_secondary": 0, "immunity": [], "resistance": [], "attack_type_primary": "Weapon",
     "attack_type_secondary": "", "big": False, "paralyzed": 0, "long_paralyzed": 0, "running_away": 0, "transformed": 0,
     "basestats": []},
]