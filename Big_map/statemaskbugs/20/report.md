# №20. Загрузка состояния без нового необязательного поля использует безопасное значение по умолчанию.

- Раздел: Сохранение состояния и маски действий
- Результат: **ОШИБКА/ПРОБЕЛ**
- Прогонов: 100
- Успешно: 0
- Неуспешно: 100
- Время: 0.00 с

Оракул: проверка наличия необходимой игровой системы/API.

## Найденное

- 100/100: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.

## Примеры seed

- seed `930000`: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.
- seed `930001`: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.
- seed `930002`: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.
- seed `930003`: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.
- seed `930004`: У CampaignEnv нет API сериализации/десериализации состояния (save/load, get_state/set_state); get_state_summary — только неполный отчёт.
