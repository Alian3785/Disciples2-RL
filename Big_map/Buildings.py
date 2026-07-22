# Империя (DII) — добавлен ключ unit в конец каждой записи
empire_buildings_d2 = {
    "alredybuilt": 0,
    "emp_d2_b001": {"id": "emp_d2_b001", "name": "Конюшня",            "gold": 2,    "requires": [],                     "blocks": ["Темница"],             "built": 0, "unit": "Рыцарь", "blocked": 0},
    "emp_d2_b002": {"id": "emp_d2_b002", "name": "Темница",            "gold": 2,    "requires": [],                     "blocks": ["Конюшня"],             "built": 0, "unit": "Охотник на ведьм", "blocked": 0},
    "emp_d2_b003": {"id": "emp_d2_b003", "name": "Имперская конюшня",  "gold": 5,    "requires": ["Конюшня"],           "blocks": [],                      "built": 0, "unit": "Имперский рыцарь", "blocked": 0},
    "emp_d2_b004": {"id": "emp_d2_b004", "name": "Пыточная камера",    "gold": 5,    "requires": ["Темница"],           "blocks": [],                      "built": 0, "unit": "Инквизитор", "blocked": 0},
    "emp_d2_b005": {"id": "emp_d2_b005", "name": "Фонтан Жизни",       "gold": 15,   "requires": ["Имперская конюшня"], "blocks": ["Святыня"],             "built": 0, "unit": "Ангел", "blocked": 0},
    "emp_d2_b006": {"id": "emp_d2_b006", "name": "Святыня",            "gold": 15,   "requires": ["Имперская конюшня"], "blocks": ["Фонтан Жизни"],        "built": 0, "unit": "Паладин", "blocked": 0},
    "emp_d2_b007": {"id": "emp_d2_b007", "name": "Магистрат",          "gold": 15,   "requires": ["Пыточная камера"],   "blocks": [],                      "built": 0, "unit": "Великий инквизитор", "blocked": 0},
    "emp_d2_b008": {"id": "emp_d2_b008", "name": "Священная статуя",   "gold": 30,   "requires": ["Святыня"],           "blocks": ["Часовня"],             "built": 0, "unit": "Мастер клинка", "blocked": 0},
    "emp_d2_b009": {"id": "emp_d2_b009", "name": "Часовня",            "gold": 30,   "requires": ["Святыня"],           "blocks": ["Священная статуя"],    "built": 0, "unit": "Защитник Веры", "blocked": 0},

    "emp_d2_b010": {"id": "emp_d2_b010", "name": "Библиотека",         "gold": 2,    "requires": [],                     "blocks": [],                      "built": 0, "unit": "Маг", "blocked": 0},
    "emp_d2_b011": {"id": "emp_d2_b011", "name": "Разрушенный храм",   "gold": 7.5,  "requires": ["Библиотека"],        "blocks": [],                      "built": 0, "unit": "Титан", "blocked": 0},
    "emp_d2_b012": {"id": "emp_d2_b012", "name": "Башня",              "gold": 5,    "requires": ["Библиотека"],        "blocks": ["Круг стихий"],         "built": 0, "unit": "Волшебник", "blocked": 0},
    "emp_d2_b013": {"id": "emp_d2_b013", "name": "Круг стихий",        "gold": 5,    "requires": ["Библиотека"],        "blocks": ["Башня"],               "built": 0, "unit": "Элементалист", "blocked": 0},
    "emp_d2_b014": {"id": "emp_d2_b014", "name": "Башня Тайн",         "gold": 15,   "requires": ["Башня"],             "blocks": [],                      "built": 0, "unit": "Белый маг", "blocked": 0},

    "emp_d2_b015": {"id": "emp_d2_b015", "name": "Стрелковый полигон", "gold": 2,    "requires": [],                     "blocks": [],                      "built": 0, "unit": "Стрелок", "blocked": 0},
    "emp_d2_b016": {"id": "emp_d2_b016", "name": "Имперская гильдия",  "gold": 5,    "requires": ["Стрелковый полигон"],"blocks": [],                      "built": 0, "unit": "Имперский ассасин", "blocked": 0},

    "emp_d2_b017": {"id": "emp_d2_b017", "name": "Монастырь",          "gold": 2,    "requires": [],                     "blocks": ["Церковь"],             "built": 0, "unit": "Жрец", "blocked": 0},
    "emp_d2_b018": {"id": "emp_d2_b018", "name": "Церковь",            "gold": 2,    "requires": [],                     "blocks": ["Монастырь"],           "built": 0, "unit": "Клирик", "blocked": 0},
    "emp_d2_b019": {"id": "emp_d2_b019", "name": "Святилище",          "gold": 5,    "requires": ["Монастырь"],         "blocks": [],                      "built": 0, "unit": "Священник", "blocked": 0},
    "emp_d2_b020": {"id": "emp_d2_b020", "name": "Собор",              "gold": 5,    "requires": ["Церковь"],           "blocks": [],                      "built": 0, "unit": "Аббатиса", "blocked": 0},
    "emp_d2_b021": {"id": "emp_d2_b021", "name": "Дом молитвы",        "gold": 15,   "requires": ["Святилище"],         "blocks": [],                      "built": 0, "unit": "Патриарх", "blocked": 0},
    "emp_d2_b022": {"id": "emp_d2_b022", "name": "Базилика",           "gold": 15,   "requires": ["Собор"],             "blocks": [],                      "built": 0, "unit": "Прорицательница", "blocked": 0},

    "emp_d2_b023": {"id": "emp_d2_b023", "name": "Гильдия",            "gold": 1.5,  "requires": [],                     "blocks": [],                      "built": 0, "unit": "Вор империи", "blocked": 0},
    "emp_d2_b024": {"id": "emp_d2_b024", "name": "Башня магии",        "gold": 1.5,  "requires": [],                     "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
    "emp_d2_b025": {"id": "emp_d2_b025", "name": "Храм",               "gold": 3,    "requires": [],                     "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
}


# Горные Кланы (DII)
mountain_clans_buildings_d2 = {
    "alredybuilt": 0,
    "mcl_d2_b001": {"id": "mcl_d2_b001", "name": "Пивоварня",          "gold": 2,    "requires": [],                    "blocks": [],                      "built": 0, "unit": "Гном воин", "blocked": 0},
    "mcl_d2_b002": {"id": "mcl_d2_b002", "name": "Арсенал",            "gold": 5,    "requires": ["Пивоварня"],         "blocks": ["Форпост"],             "built": 0, "unit": "Ветеран", "blocked": 0},
    "mcl_d2_b003": {"id": "mcl_d2_b003", "name": "Форпост",            "gold": 5,    "requires": ["Пивоварня"],         "blocks": ["Арсенал"],             "built": 0, "unit": "Горец", "blocked": 0},
    "mcl_d2_b004": {"id": "mcl_d2_b004", "name": "Святыня Предков",    "gold": 15,   "requires": ["Арсенал"],           "blocks": [],                      "built": 0, "unit": "Старый ветеран", "blocked": 0},
    "mcl_d2_b005": {"id": "mcl_d2_b005", "name": "Снежная хижина",     "gold": 15,   "requires": ["Форпост"],           "blocks": ["Чертог Фенрира"],      "built": 0, "unit": "Отшельник", "blocked": 0},
    "mcl_d2_b006": {"id": "mcl_d2_b006", "name": "Чертог Фенрира",     "gold": 15,   "requires": ["Форпост"],           "blocks": ["Снежная хижина"],      "built": 0, "unit": "Повелитель волков", "blocked": 0},
    "mcl_d2_b007": {"id": "mcl_d2_b007", "name": "Чертог короля",      "gold": 45,   "requires": ["Святыня Предков"],   "blocks": ["Зал рун"],             "built": 0, "unit": "Король гномов", "blocked": 0},
    "mcl_d2_b008": {"id": "mcl_d2_b008", "name": "Зал рун",            "gold": 30,   "requires": ["Святыня Предков"],   "blocks": ["Чертог короля"],       "built": 0, "unit": "Владыка рун", "blocked": 0},

    "mcl_d2_b009": {"id": "mcl_d2_b009", "name": "Домик в горах",      "gold": 2,    "requires": [],                    "blocks": [],                      "built": 0, "unit": "Посвященная", "blocked": 0},
    "mcl_d2_b010": {"id": "mcl_d2_b010", "name": "Горное логово",      "gold": 7.5,  "requires": ["Домик в горах"],     "blocks": [],                      "built": 0, "unit": "Йети", "blocked": 0},
    "mcl_d2_b011": {"id": "mcl_d2_b011", "name": "Сад Мудрости",       "gold": 5,    "requires": ["Домик в горах"],     "blocks": ["Башня алхимии"],       "built": 0, "unit": "Друид", "blocked": 0},
    "mcl_d2_b012": {"id": "mcl_d2_b012", "name": "Башня алхимии",      "gold": 5,    "requires": ["Домик в горах"],     "blocks": ["Сад Мудрости"],        "built": 0, "unit": "Алхимик", "blocked": 0},
    "mcl_d2_b013": {"id": "mcl_d2_b013", "name": "Бузина",             "gold": 15,   "requires": ["Сад Мудрости"],      "blocks": [],                      "built": 0, "unit": "Архидруид", "blocked": 0},

    "mcl_d2_b014": {"id": "mcl_d2_b014", "name": "Стрельбище",         "gold": 2,    "requires": [],                    "blocks": [],                      "built": 0, "unit": "Арбалетчик", "blocked": 0},
    "mcl_d2_b015": {"id": "mcl_d2_b015", "name": "Незатухающая кузня", "gold": 5,    "requires": ["Стрельбище"],        "blocks": ["Гильдия инженеров"],   "built": 0, "unit": "Хранитель кузни", "blocked": 0},
    "mcl_d2_b016": {"id": "mcl_d2_b016", "name": "Гильдия инженеров",  "gold": 7,    "requires": ["Стрельбище"],        "blocks": ["Незатухающая кузня"],  "built": 0, "unit": "Огнеметчик", "blocked": 0},

    "mcl_d2_b017": {"id": "mcl_d2_b017", "name": "Горный пик",         "gold": 3,    "requires": [],                    "blocks": [],                      "built": 0, "unit": "Скальной гигант", "blocked": 0},
    "mcl_d2_b018": {"id": "mcl_d2_b018", "name": "Облачная крепость",  "gold": 7.5,  "requires": ["Горный пик"],        "blocks": ["Ледяные горы"],        "built": 0, "unit": "Штормовой гигант", "blocked": 0},
    "mcl_d2_b019": {"id": "mcl_d2_b019", "name": "Ледяные горы",       "gold": 7.5,  "requires": ["Горный пик"],        "blocks": ["Облачная крепость"],   "built": 0, "unit": "Ледяной гигант", "blocked": 0},
    "mcl_d2_b020": {"id": "mcl_d2_b020", "name": "Мост Биврёст",       "gold": 22.5, "requires": ["Облачная крепость"], "blocks": [],                      "built": 0, "unit": "Старейшина", "blocked": 0},
    "mcl_d2_b021": {"id": "mcl_d2_b021", "name": "Ледяные пещеры",     "gold": 22.5, "requires": ["Ледяные горы"],      "blocks": [],                      "built": 0, "unit": "Сын Имира", "blocked": 0},

    "mcl_d2_b022": {"id": "mcl_d2_b022", "name": "Гильдия",            "gold": 1.5,  "requires": [],                    "blocks": [],                      "built": 0, "unit": "Вор гномов", "blocked": 0},
    "mcl_d2_b023": {"id": "mcl_d2_b023", "name": "Башня магии",        "gold": 1.5,  "requires": [],                    "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
    "mcl_d2_b024": {"id": "mcl_d2_b024", "name": "Храм",               "gold": 3,    "requires": [],                    "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
}


# Орды Нежити (DII)
undead_hordes_buildings_d2 = {
    "alredybuilt": 0,
    "und_d2_b001": {"id": "und_d2_b001", "name": "Проклятая земля",     "gold": 2,    "requires": [],                      "blocks": ["Зловещий монастырь"], "built": 0, "unit": "Зомби", "blocked": 0},
    "und_d2_b002": {"id": "und_d2_b002", "name": "Зловещий монастырь",  "gold": 2,    "requires": [],                      "blocks": ["Проклятая земля"],    "built": 0, "unit": "Тамплиер", "blocked": 0},
    "und_d2_b003": {"id": "und_d2_b003", "name": "Кладбище",           "gold": 5,    "requires": ["Проклятая земля"],     "blocks": [],                      "built": 0, "unit": "Скелет воин", "blocked": 0},
    "und_d2_b004": {"id": "und_d2_b004", "name": "Тёмный идол",        "gold": 5,    "requires": ["Зловещий монастырь"],  "blocks": [],                      "built": 0, "unit": "Лорд тьмы", "blocked": 0},
    "und_d2_b005": {"id": "und_d2_b005", "name": "Хранилище душ",      "gold": 15,   "requires": ["Кладбище"],            "blocks": [],                      "built": 0, "unit": "Скелет рыцарь", "blocked": 0},
    "und_d2_b006": {"id": "und_d2_b006", "name": "Крематорий",         "gold": 30,   "requires": ["Хранилище душ"],       "blocks": [],                      "built": 0, "unit": "Скелет призрак", "blocked": 0},

    "und_d2_b007": {"id": "und_d2_b007", "name": "Тёмный храм",        "gold": 2,    "requires": [],                      "blocks": [],                      "built": 0, "unit": "Чернокнижник", "blocked": 0},
    "und_d2_b008": {"id": "und_d2_b008", "name": "Оккультный храм",    "gold": 5,    "requires": ["Тёмный храм"],         "blocks": ["Проклятая часовня"],   "built": 0, "unit": "Некромант", "blocked": 0},
    "und_d2_b009": {"id": "und_d2_b009", "name": "Проклятая часовня",  "gold": 6.2,  "requires": ["Тёмный храм"],         "blocks": ["Оккультный храм"],     "built": 0, "unit": "Дух", "blocked": 0},
    "und_d2_b010": {"id": "und_d2_b010", "name": "Крипта",             "gold": 15,   "requires": ["Оккультный храм"],     "blocks": ["Тёмная башня"],        "built": 0, "unit": "Вампир", "blocked": 0},
    "und_d2_b011": {"id": "und_d2_b011", "name": "Тёмная башня",       "gold": 15,   "requires": ["Оккультный храм"],     "blocks": ["Крипта"],              "built": 0, "unit": "Лич", "blocked": 0},
    "und_d2_b012": {"id": "und_d2_b012", "name": "Обитель Мортис",     "gold": 18.75,"requires": ["Проклятая часовня"],   "blocks": ["Ловушка душ"],         "built": 0, "unit": "Смерть", "blocked": 0},
    "und_d2_b013": {"id": "und_d2_b013", "name": "Ловушка душ",        "gold": 22.5, "requires": ["Проклятая часовня"],   "blocks": ["Обитель Мортис"],      "built": 0, "unit": "Сущий", "blocked": 0},
    "und_d2_b014": {"id": "und_d2_b014", "name": "Цитадель",           "gold": 30,   "requires": ["Крипта"],              "blocks": [],                      "built": 0, "unit": "Высший вампир", "blocked": 0},
    "und_d2_b015": {"id": "und_d2_b015", "name": "Башня колдовства",   "gold": 30,   "requires": ["Тёмная башня"],        "blocks": [],                      "built": 0, "unit": "Архилич", "blocked": 0},

    "und_d2_b016": {"id": "und_d2_b016", "name": "Гробница",           "gold": 2,    "requires": [],                      "blocks": [],                      "built": 0, "unit": "Призрак", "blocked": 0},
    "und_d2_b017": {"id": "und_d2_b017", "name": "Катакомбы",          "gold": 10,   "requires": ["Гробница"],            "blocks": [],                      "built": 0, "unit": "Тень", "blocked": 0},

    "und_d2_b018": {"id": "und_d2_b018", "name": "Пещера",             "gold": 3,    "requires": [],                      "blocks": [],                      "built": 0, "unit": "Дракон рока", "blocked": 0},
    "und_d2_b019": {"id": "und_d2_b019", "name": "Кладбище костей",    "gold": 7.5,  "requires": ["Пещера"],              "blocks": [],                      "built": 0, "unit": "Дракон смерти", "blocked": 0},
    "und_d2_b020": {"id": "und_d2_b020", "name": "Логово оборотней",   "gold": 7.5,  "requires": ["Пещера"],              "blocks": [],                      "built": 0, "unit": "Оборотень", "blocked": 0},
    "und_d2_b021": {"id": "und_d2_b021", "name": "Проклятый могильник","gold": 22.5, "requires": ["Кладбище костей"],     "blocks": ["Драконий могильник"],  "built": 0, "unit": "Змей ужаса", "blocked": 0},
    "und_d2_b022": {"id": "und_d2_b022", "name": "Драконий могильник", "gold": 22.5, "requires": ["Кладбище костей"],     "blocks": ["Проклятый могильник"], "built": 0, "unit": "Драколич", "blocked": 0},

    "und_d2_b023": {"id": "und_d2_b023", "name": "Гильдия",            "gold": 1.5,  "requires": [],                      "blocks": [],                      "built": 0, "unit": "Вор нежити", "blocked": 0},
    "und_d2_b024": {"id": "und_d2_b024", "name": "Башня магии",        "gold": 1.5,  "requires": [],                      "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
    "und_d2_b025": {"id": "und_d2_b025", "name": "Храм",               "gold": 3,    "requires": [],                      "blocks": [],                      "built": 0, "unit": None, "blocked": 0},
}


# Легионы Проклятых (DII)
legions_buildings_d2 = {
    "alredybuilt": 0,
    "lod_d2_b001": {"id": "lod_d2_b001", "name": "Нечестивый портал",     "gold": 2,    "requires": [],                     "blocks": [],                     "built": 0, "unit": "Берсерк", "blocked": 0},
    "lod_d2_b002": {"id": "lod_d2_b002", "name": "Храм Скорби",          "gold": 7.5,  "requires": ["Нечестивый портал"],   "blocks": [],                     "built": 0, "unit": "Сатир", "blocked": 0},
    "lod_d2_b003": {"id": "lod_d2_b003", "name": "Башня душ",            "gold": 5,    "requires": ["Нечестивый портал"],   "blocks": [],                     "built": 0, "unit": "Темный паладин", "blocked": 0},
    "lod_d2_b004": {"id": "lod_d2_b004", "name": "Идол Бетрезена",       "gold": 15,   "requires": ["Башня душ"],           "blocks": [],                     "built": 0, "unit": "Адский рыцарь", "blocked": 0},

    "lod_d2_b005": {"id": "lod_d2_b005", "name": "Аббатство оккультизма","gold": 2,    "requires": [],                     "blocks": ["Дворец Порока"],      "built": 0, "unit": "Колдун", "blocked": 0},
    "lod_d2_b006": {"id": "lod_d2_b006", "name": "Дворец Порока",        "gold": 2,    "requires": [],                     "blocks": ["Аббатство оккультизма"],"built": 0,"unit": "Ведьма", "blocked": 0},
    "lod_d2_b007": {"id": "lod_d2_b007", "name": "Крепость душ",         "gold": 5,    "requires": ["Аббатство оккультизма"],"blocks": ["Тёмное святилище"],  "built": 0, "unit": "Двойник", "blocked": 0},
    "lod_d2_b008": {"id": "lod_d2_b008", "name": "Тёмное святилище",     "gold": 5,    "requires": ["Аббатство оккультизма"],"blocks": ["Крепость душ"],      "built": 0, "unit": "Демонолог", "blocked": 0},
    "lod_d2_b009": {"id": "lod_d2_b009", "name": "Дворец Греха",         "gold": 5,    "requires": ["Дворец Порока"],       "blocks": [],                     "built": 0, "unit": "Колдунья", "blocked": 0},
    "lod_d2_b010": {"id": "lod_d2_b010", "name": "Зал Обмана",           "gold": 15,   "requires": ["Тёмное святилище"],    "blocks": ["Алтарь"],             "built": 0, "unit": "Инкуб", "blocked": 0},
    "lod_d2_b011": {"id": "lod_d2_b011", "name": "Алтарь",               "gold": 15,   "requires": ["Тёмное святилище"],    "blocks": ["Зал Обмана"],         "built": 0, "unit": "Пандемонеус", "blocked": 0},
    "lod_d2_b012": {"id": "lod_d2_b012", "name": "Высокий храм",         "gold": 15,   "requires": ["Дворец Греха"],          "blocks": [],                     "built": 0, "unit": "Суккуб", "blocked": 0},
    "lod_d2_b013": {"id": "lod_d2_b013", "name": "Святыня поклонения",   "gold": 30,   "requires": ["Алтарь"],             "blocks": [],                     "built": 0, "unit": "Модеус", "blocked": 0},

    "lod_d2_b014": {"id": "lod_d2_b014", "name": "Шпиль",                "gold": 3,    "requires": [],                     "blocks": [],                     "built": 0, "unit": "Мраморная гаргулья", "blocked": 0},
    "lod_d2_b015": {"id": "lod_d2_b015", "name": "Ониксовый шпиль",      "gold": 7.5,  "requires": ["Шпиль"],              "blocks": [],                     "built": 0, "unit": "Ониксовая гаргулья", "blocked": 0},

    "lod_d2_b016": {"id": "lod_d2_b016", "name": "Нечестивые врата",     "gold": 3,    "requires": [],                     "blocks": [],                     "built": 0, "unit": "Демон", "blocked": 0},
    "lod_d2_b017": {"id": "lod_d2_b017", "name": "Часовня Истязаний",    "gold": 7.5,  "requires": ["Нечестивые врата"],    "blocks": [],                     "built": 0, "unit": "Молох", "blocked": 0},
    "lod_d2_b018": {"id": "lod_d2_b018", "name": "Нечестивый алтарь",    "gold": 22.5, "requires": ["Часовня Истязаний"],   "blocks": ["Адский портал"],      "built": 0, "unit": "Тварь", "blocked": 0},
    "lod_d2_b019": {"id": "lod_d2_b019", "name": "Адский портал",        "gold": 22.5, "requires": ["Часовня Истязаний"],   "blocks": ["Нечестивый алтарь"],  "built": 0, "unit": "Повелитель демонов", "blocked": 0},
    "lod_d2_b020": {"id": "lod_d2_b020", "name": "Темница с рабами",     "gold": 45,   "requires": ["Нечестивый алтарь"],   "blocks": [],                     "built": 0, "unit": "Тиамат", "blocked": 0},
    "lod_d2_b021": {"id": "lod_d2_b021", "name": "Нечестивые залы",      "gold": 45,   "requires": ["Адский портал"],       "blocks": ["Нечестивый особняк"], "built": 0, "unit": "Владыка", "blocked": 0},
    "lod_d2_b022": {"id": "lod_d2_b022", "name": "Нечестивый особняк",   "gold": 45,   "requires": ["Адский портал"],       "blocks": ["Нечестивые залы"],    "built": 0, "unit": "Дьявол Бездны", "blocked": 0},

    "lod_d2_b023": {"id": "lod_d2_b023", "name": "Гильдия",              "gold": 1.5,  "requires": [],                     "blocks": [],                     "built": 0, "unit": "Вор легионов", "blocked": 0},
    "lod_d2_b024": {"id": "lod_d2_b024", "name": "Башня магии",          "gold": 1.5,  "requires": [],                     "blocks": [],                     "built": 0, "unit": None, "blocked": 0},
    "lod_d2_b025": {"id": "lod_d2_b025", "name": "Храм",                 "gold": 3,    "requires": [],                     "blocks": [],                     "built": 0, "unit": None, "blocked": 0},
}


# Альянс Эльфов (DII)
elves_buildings_d2 = {
    "alredybuilt": 0,
    "elf_d2_b001": {"id": "elf_d2_b001", "name": "Пещера кентавров",   "gold": 2,    "requires": [],                   "blocks": ["Стойла кентавров"],   "built": 0, "unit": "Кентавр странник", "blocked": 0},
    "elf_d2_b002": {"id": "elf_d2_b002", "name": "Стойла кентавров",   "gold": 2,    "requires": [],                   "blocks": ["Пещера кентавров"],   "built": 0, "unit": "Кентавр латник", "blocked": 0},
    "elf_d2_b003": {"id": "elf_d2_b003", "name": "Земли племён",       "gold": 5,    "requires": ["Пещера кентавров"], "blocks": [],                     "built": 0, "unit": "Кентавр дикарь", "blocked": 0},

    "elf_d2_b004": {"id": "elf_d2_b004", "name": "Глава",              "gold": 2,    "requires": [],                   "blocks": [],                     "built": 0, "unit": "Проводник", "blocked": 0},
    "elf_d2_b005": {"id": "elf_d2_b005", "name": "Птичник",            "gold": 7.5,  "requires": ["Глава"],            "blocks": [],                     "built": 0, "unit": "Грифон", "blocked": 0},
    "elf_d2_b006": {"id": "elf_d2_b006", "name": "Воларий",            "gold": 15,   "requires": ["Птичник"],          "blocks": [],                     "built": 0, "unit": "Повелитель небес", "blocked": 0},
    "elf_d2_b007": {"id": "elf_d2_b007", "name": "Элементариум",       "gold": 5,    "requires": ["Глава"],            "blocks": ["Верховный совет"],    "built": 0, "unit": "Теург", "blocked": 0},
    "elf_d2_b008": {"id": "elf_d2_b008", "name": "Верховный совет",    "gold": 5,    "requires": ["Глава"],            "blocks": ["Элементариум"],       "built": 0, "unit": "Архонт", "blocked": 0},

    "elf_d2_b009": {"id": "elf_d2_b009", "name": "Хижина скорняка",    "gold": 2,    "requires": [],                   "blocks": ["Дозорная башня"],     "built": 0, "unit": "Охотник", "blocked": 0},
    "elf_d2_b010": {"id": "elf_d2_b010", "name": "Дозорная башня",     "gold": 2,    "requires": [],                   "blocks": ["Хижина скорняка"],    "built": 0, "unit": "Дозорный", "blocked": 0},
    "elf_d2_b011": {"id": "elf_d2_b011", "name": "Лачуга",             "gold": 5,    "requires": ["Хижина скорняка"],  "blocks": ["Хибара"],             "built": 0, "unit": "Бандит", "blocked": 0},
    "elf_d2_b012": {"id": "elf_d2_b012", "name": "Хибара",             "gold": 5,    "requires": ["Хижина скорняка"],  "blocks": ["Лачуга"],             "built": 0, "unit": "Стингер", "blocked": 0},
    "elf_d2_b013": {"id": "elf_d2_b013", "name": "Пиромантиум",        "gold": 5,    "requires": ["Дозорная башня"],   "blocks": ["Гидромантиум"],       "built": 0, "unit": "Смотритель", "blocked": 0},
    "elf_d2_b014": {"id": "elf_d2_b014", "name": "Гидромантиум",       "gold": 5,    "requires": ["Дозорная башня"],   "blocks": ["Пиромантиум"],        "built": 0, "unit": "Часовой", "blocked": 0},
    "elf_d2_b015": {"id": "elf_d2_b015", "name": "Укрытие",            "gold": 15,   "requires": ["Лачуга"],           "blocks": [],                     "built": 0, "unit": "Разбойник эльф", "blocked": 0},
    "elf_d2_b016": {"id": "elf_d2_b016", "name": "Пристанище",         "gold": 30,   "requires": ["Укрытие"],          "blocks": [],                     "built": 0, "unit": "Мародёр", "blocked": 0},

    "elf_d2_b017": {"id": "elf_d2_b017", "name": "Круг спригганов",    "gold": 2,    "requires": [],                   "blocks": [],                     "built": 0, "unit": "Оракул", "blocked": 0},
    "elf_d2_b018": {"id": "elf_d2_b018", "name": "Заросли",            "gold": 5,    "requires": ["Круг спригганов"],  "blocks": [],                     "built": 0, "unit": "Дева рощи", "blocked": 0},
    "elf_d2_b019": {"id": "elf_d2_b019", "name": "Огненная земля",     "gold": 15,   "requires": ["Заросли"],          "blocks": ["Алтарь Галлеана"],    "built": 0, "unit": "Солнечная танцовщица", "blocked": 0},
    "elf_d2_b020": {"id": "elf_d2_b020", "name": "Алтарь Галлеана",    "gold": 15,   "requires": ["Заросли"],          "blocks": ["Огненная земля"],     "built": 0, "unit": "Сильфида", "blocked": 0},

    "elf_d2_b021": {"id": "elf_d2_b021", "name": "Гильдия",            "gold": 1.5,  "requires": [],                   "blocks": [],                     "built": 0, "unit": "Вор эльфов", "blocked": 0},
    "elf_d2_b022": {"id": "elf_d2_b022", "name": "Башня магии",        "gold": 1.5,  "requires": [],                   "blocks": [],                     "built": 0, "unit": None, "blocked": 0},
    "elf_d2_b023": {"id": "elf_d2_b023", "name": "Храм",               "gold": 3,    "requires": [],                   "blocks": [],                     "built": 0, "unit": None, "blocked": 0},
}


_BUILDING_TABLES_D2 = {
    "empire": empire_buildings_d2,
    "mountain_clans": mountain_clans_buildings_d2,
    "undead_hordes": undead_hordes_buildings_d2,
    "legions": legions_buildings_d2,
    "elves": elves_buildings_d2,
}


def _building_name_tuple(raw_names):
    if isinstance(raw_names, str):
        names_iter = (raw_names,)
    elif isinstance(raw_names, (list, tuple, set)):
        names_iter = raw_names
    else:
        names_iter = ()
    return tuple(
        str(name).strip()
        for name in names_iter
        if str(name).strip()
    )


def _normalize_building_table(table):
    for entry in table.values():
        if not isinstance(entry, dict):
            continue
        entry["requires"] = _building_name_tuple(entry.get("requires", ()))
        entry["blocks"] = _building_name_tuple(entry.get("blocks", ()))


def _build_building_metadata(table):
    keys = []
    key_by_name = {}
    metadata_by_key = {}
    max_gold = 1.0

    for key, entry in table.items():
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        key_s = str(key)
        keys.append(key_s)
        name = str(entry.get("name", "") or "").strip()
        if name:
            key_by_name[name] = key_s
        try:
            max_gold = max(max_gold, float(entry.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

    for key_s in keys:
        entry = table[key_s]
        requirements = _building_name_tuple(entry.get("requires", ()))
        blocks = _building_name_tuple(entry.get("blocks", ()))
        try:
            base_gold = max(0.0, float(entry.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            base_gold = 0.0
        metadata_by_key[key_s] = {
            "name": str(entry.get("name", "") or "").strip(),
            "requires": requirements,
            "blocks": blocks,
            "requirement_keys": tuple(
                key_by_name[name]
                for name in requirements
                if name in key_by_name
            ),
            "block_keys": tuple(
                key_by_name[name]
                for name in blocks
                if name in key_by_name
            ),
            "base_gold": base_gold,
        }

    return {
        "keys": tuple(keys),
        "key_by_name": dict(key_by_name),
        "metadata_by_key": metadata_by_key,
        "max_gold": float(max_gold),
    }


for _building_table in _BUILDING_TABLES_D2.values():
    _normalize_building_table(_building_table)


BUILDING_METADATA_D2 = {
    key: _build_building_metadata(table)
    for key, table in _BUILDING_TABLES_D2.items()
}
