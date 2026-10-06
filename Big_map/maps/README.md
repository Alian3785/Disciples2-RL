# Схемы всех карт

Для каждой зарегистрированной карты схема находится в `<имя_карты>/layout.png`.
В соседних `README.md` и `layout.json` сохранены составы армий, координаты,
предметы, ресурсы и действующие клетки взаимодействия с объектами.
Для `city_defence_train`, `siege_train` и `super_last_stand` добавлен `waves.png`.

Схемы получены из реальной среды после `reset(seed=0)` с `local5`,
`use_boss_starting_roster=False` и `scripted_capital_bot_enabled=False`.
Закреплённые сценарием стартовые отряды сохраняются. Будущие волны отделены
от активных отрядов. На карте показаны фактические игровые координаты после
масштабирования и разрешения конфликтов. Большие существа считаются один раз.

## Все 16 карт

- [Builder](builder/README.md)
- [City Defence](city_defence_train/README.md)
- [A Return to Simpler Times](default/README.md)
- [Formation](formation_train/README.md)
- [Green Dragon Minimal](green_dragon_minimal/README.md)
- [Hire](hire_train/README.md)
- [Item](item_train/README.md)
- [Magic](magic_train/README.md)
- [Mirrow match](mirrow_match/README.md)
- [Orc Duel](orc_duel/README.md)
- [Scroll](scroll_train/README.md)
- [Siege](siege_train/README.md)
- [Small](small/README.md)
- [Super Last Stand](super_last_stand/README.md)
- [Trade](trade_train/README.md)
- [Wotan’s Retribution](wotans_retribution/README.md)

## Перегенерация

Из папки `Big_map`, в окружении с зависимостями игры и Pillow:

```sh
python tools/generate_map_layouts.py
python tools/generate_map_layouts.py --map mirrow_match
python -m pytest -q tests/test_map_layout_documentation.py
```

Исходные модули `maps/<имя>.py` не перемещены. Папки со схемами не содержат
`__init__.py` и не подменяют игровые импорты. Генератор не запускает обучение,
оценку или replay и не меняет игровую логику.
