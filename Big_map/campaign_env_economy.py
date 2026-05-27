# campaign_env_economy.py
"""CampaignEconomyMixin for CampaignEnv."""

from __future__ import annotations

from campaign_env_data import *


class CampaignEconomyMixin:
    """Экономические и ростерные действия CampaignEnv.

    Миксин связывает карту кампании с состоянием синего отряда: лечение,
    воскрешение, найм, здания, очки хода героя и перестановку юнитов в
    боевых позициях.
    """

    @staticmethod
    def _castle_heal_gold_per_hp(unit: Dict) -> float:
        """Вернуть цену восстановления одного HP для юнита на клетке лечения.

        Фракционные юниты тарифицируются по уровню, потому что их сила уже
        привязана к ветке прокачки. Нейтралы не всегда имеют корректный Level,
        поэтому для них используется bucket по максимальному здоровью.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 1.0
            if max_hp <= 250.0:
                return 2.0
            if max_hp <= 500.0:
                return 3.0
            return 4.0

        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level in (2, 3):
            return 2.0
        if level >= 4:
            return 3.0
        return 1.0
    @staticmethod
    def _castle_revive_gold_cost(unit: Dict) -> float:
        """Посчитать полную стоимость воскрешения юнита золотом.

        Для обычных фракционных юнитов цена растёт по Level. Для нейтралов
        используется шкала max HP: такие существа могут быть сильными без
        нормальной фракционной ветки, и Level для них ненадёжен.
        """
        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False

        if is_neutral_unit:
            max_hp_raw = unit.get("maxhp", unit.get("max_health", 0))
            try:
                max_hp = float(max_hp_raw or 0.0)
            except (TypeError, ValueError):
                max_hp = 0.0

            if max_hp <= 100.0:
                return 50.0
            if max_hp <= 150.0:
                return 200.0
            if max_hp <= 250.0:
                return 400.0
            if max_hp <= 350.0:
                return 600.0
            if max_hp <= 500.0:
                return 800.0
            if max_hp <= 2000.0:
                return 1000.0
            return 1500.0

        level_raw = unit.get("Level", 1)
        try:
            level = int(round(float(level_raw)))
        except (TypeError, ValueError):
            level = 1

        if level <= 1:
            return 50.0
        if level == 2:
            return 200.0
        if level == 3:
            return 400.0
        if level == 4:
            return 600.0
        return 800.0
    @classmethod
    def _trainer_gold_per_xp(cls, unit: Dict) -> float:
        """Вернуть цену одного очка опыта у тренера для выбранного юнита.

        Герои и нейтралы получают фиксированный дорогой тариф. Обычные
        фракционные юниты масштабируются по текущему уровню, чтобы покупка
        опыта становилась дороже вместе с ростом силы отряда.
        """
        if not isinstance(unit, dict):
            return 0.0

        is_neutral_unit = bool(unit.get("is_neutral_unit", False))
        if not is_neutral_unit:
            capital_raw = unit.get("capital", None)
            try:
                is_neutral_unit = capital_raw is not None and int(round(float(capital_raw))) == 0
            except (TypeError, ValueError):
                is_neutral_unit = False
        if is_neutral_unit or cls._is_hero_unit(unit):
            return 5.0

        level_raw = unit.get("Level", 0)
        try:
            level = int(round(float(level_raw or 0)))
        except (TypeError, ValueError):
            level = 0

        if level <= 0:
            return 5.0
        if level == 1:
            return 4.0
        if level == 2:
            return 5.0
        if level == 3:
            return 6.0
        return 7.0
    @staticmethod
    def _trainer_exp_cap(unit: Dict) -> Optional[int]:
        """Определить максимум опыта, который можно купить до level-up.

        Тренер не должен переводить юнита через следующий уровень напрямую,
        поэтому cap равен `exp_required - 1`. None означает, что у записи нет
        пригодного лимита: максимальный уровень, некорректные данные или ноль.
        """
        if not isinstance(unit, dict):
            return None
        exp_required_raw = unit.get("exp_required", 0)
        if isinstance(exp_required_raw, str) and exp_required_raw.strip().lower() == "max":
            return None
        try:
            exp_required = int(round(float(exp_required_raw or 0)))
        except (TypeError, ValueError):
            return None
        if exp_required <= 0:
            return None
        return max(0, exp_required - 1)
    @staticmethod
    def _is_hero_unit(unit: Dict) -> bool:
        """Проверить, является ли запись отряда героем кампании.

        Сначала читается явный флаг `hero`, включая строковые значения из
        таблиц. Если флага нет, fallback смотрит на `next_level_exp`, который
        обычно задан у героя и отсутствует у обычных солдат.
        """
        if not isinstance(unit, dict):
            return False
        if "hero" in unit:
            value = unit.get("hero")
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return int(value) != 0
            return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        try:
            return int(round(float(unit.get("next_level_exp", 0) or 0))) > 0
        except (TypeError, ValueError):
            return False
    def _resolve_hero_needaunit(self, unit: Dict) -> int:
        """Нормализовать флаг обязательного найма спутника для героя.

        Сценарные данные могут хранить `needaunit` как строку или число.
        Внутренний код использует строго 0/1, потому что флаг влияет на штрафы
        за ход и сбрасывается после успешного фракционного найма.
        """
        try:
            return 1 if int(round(float(unit.get("needaunit", 0) or 0))) > 0 else 0
        except (TypeError, ValueError):
            return 0
    @classmethod
    def _hero_display_name(cls, unit: Optional[Dict]) -> str:
        """Вернуть безопасное отображаемое имя героя или юнита.

        Основной ключ `name` используется в нормализованных данных, а старый
        ключ оставлен fallback'ом для legacy-таблиц. Невалидные записи дают
        пустую строку, чтобы проверки по имени не падали.
        """
        if not isinstance(unit, dict):
            return ""
        return str(unit.get("name", unit.get("кто", "")) or "").strip()
    @staticmethod
    def _truthy_unit_flag(unit: Dict, *keys: str) -> bool:
        """Проверить набор булевых флагов юнита в разных форматах.

        Поля из данных могут быть bool, числом или строкой вроде `true`.
        Helper перебирает ключи и возвращает True при первом явно истинном
        значении, скрывая форматирование источника от вызывающего кода.
        """
        for key in keys:
            if key not in unit:
                continue
            value = unit.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return int(value) != 0
            if str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                return True
        return False
    def _hero_has_flying(self, hero: Optional[Dict]) -> bool:
        """Определить, умеет ли герой летать на карте кампании.

        Проверяются явные флаги, token в `hero_abilities` и список имён героев,
        которым полёт задан сценарием. Единый helper нужен, чтобы movement,
        observation и action mask трактовали способность одинаково.
        """
        if not isinstance(hero, dict):
            return False
        if self._truthy_unit_flag(hero, "flying", "can_fly", "flight"):
            return True
        if str(self.HERO_FLYING_ABILITY_KEY).lower() in self._hero_ability_tokens(hero):
            return True
        return self._hero_display_name(hero) in set(self.HERO_FLYING_NAMES)
    def _ensure_hero_flying_ability_token(self, hero: Dict) -> None:
        """Дописать канонический token полёта известным летающим героям.

        Некоторые герои распознаются как летающие по имени. Функция синхронизирует
        это с `flying` и `hero_abilities`, не дублируя token и сохраняя удобный
        формат контейнера, если он уже был list, dict или строкой.
        """
        if not isinstance(hero, dict) or not self._is_hero_unit(hero):
            return
        if self._hero_display_name(hero) not in set(self.HERO_FLYING_NAMES):
            return

        hero["flying"] = True
        ability_key = str(self.HERO_FLYING_ABILITY_KEY)
        raw_tokens = hero.get("hero_abilities", [])
        if isinstance(raw_tokens, dict):
            raw_tokens[ability_key] = True
            hero["hero_abilities"] = raw_tokens
        elif isinstance(raw_tokens, str):
            tokens = raw_tokens.replace(",", " ").replace(";", " ").split()
            if ability_key.lower() not in {str(token).strip().lower() for token in tokens}:
                tokens.append(ability_key)
            hero["hero_abilities"] = tokens
        elif isinstance(raw_tokens, (list, tuple, set)):
            tokens = list(raw_tokens)
            if ability_key.lower() not in {str(token).strip().lower() for token in tokens}:
                tokens.append(ability_key)
            hero["hero_abilities"] = tokens
        else:
            hero["hero_abilities"] = [ability_key]
    def _sync_hero_progression_flags(self, units: Optional[List[Dict]] = None) -> None:
        """Обновить производные hero-флаги у указанного или текущего синего отряда.

        Вызывается после создания roster, найма, перестановок и других мутаций.
        Она приводит `hero`, `needaunit` и способность полёта к каноническому
        виду, чтобы экономика, observation и mask читали согласованное состояние.
        """
        roster = units if units is not None else self.blue_team_state
        if not roster:
            return
        for unit in roster:
            unit["hero"] = self._is_hero_unit(unit)
            unit["needaunit"] = self._resolve_hero_needaunit(unit)
            self._ensure_hero_flying_ability_token(unit)
    @staticmethod
    def _is_travel_unit_alive(unit: Dict) -> bool:
        """Проверить, считается ли юнит живым для логики путешествия.

        Если HP-полей нет, запись считается живой ради совместимости со старыми
        шаблонами. Если здоровье задано, требуется положительное значение хотя
        бы по одному поддерживаемому ключу.
        """
        if not isinstance(unit, dict):
            return False
        hp_keys = ("health", "hp")
        if not any(key in unit for key in hp_keys):
            return True
        for key in hp_keys:
            if key not in unit:
                continue
            try:
                return float(unit.get(key, 0) or 0) > 0.0
            except (TypeError, ValueError):
                return False
        return False
    def _resolve_travel_hero(
        self,
        units: Optional[List[Dict]] = None,
        *,
        alive_only: bool = False,
    ) -> Optional[Dict]:
        """Найти героя, который задаёт свойства путешествия отряда.

        В кампании ожидается один основной герой в синем roster. При
        `alive_only=True` мёртвый герой игнорируется, что полезно для проверок
        движения и восстановления после поражений.
        """
        roster = units if units is not None else self._get_blue_state()
        heroes = [
            unit
            for unit in roster
            if self._is_hero_unit(unit)
            and (not alive_only or self._is_travel_unit_alive(unit))
        ]
        if not heroes:
            return None
        return heroes[0]
    def _hero_base_moves(self, units: Optional[List[Dict]] = None) -> int:
        """Вернуть базовое число очков хода, которое даёт текущий герой.

        Сначала читаются явные поля движения из записи героя. Если они не заданы,
        используется таблица по имени героя, а затем общий `MOVES_PER_TURN`.
        Это позволяет сценарным героям иметь индивидуальную мобильность.
        """
        hero = self._resolve_travel_hero(units=units, alive_only=True)
        if hero is None:
            return int(self.MOVES_PER_TURN)

        for key in ("move_points", "movement", "MOVE"):
            if key not in hero:
                continue
            try:
                move_value = int(round(float(hero.get(key, 0) or 0)))
            except (TypeError, ValueError):
                move_value = 0
            if move_value > 0:
                return move_value

        hero_name = str(hero.get("name", hero.get("кто", "")) or "").strip()
        return int(self.HERO_BASE_MOVES_BY_NAME.get(hero_name, self.MOVES_PER_TURN))
    def _hero_level(self, units: Optional[List[Dict]] = None) -> int:
        """Безопасно получить уровень героя, влияющий на campaign-способности.

        Уровень используется для открытия pathfinding, leadership, lore и других
        экономических/стратегических возможностей. Невалидные значения приводятся
        к нулю, чтобы битые данные не ломали action mask.
        """
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return 0
        try:
            return max(0, int(round(float(hero.get("Level", 0) or 0))))
        except (TypeError, ValueError):
            return 0
    @classmethod
    def _hero_ability_tokens(cls, hero: Optional[Dict]) -> set[str]:
        """Нормализовать набор ability-token'ов героя в lowercase set.

        Поддерживаются строки, списки, tuple/set и dict вида `token -> enabled`.
        Возвращаемый set используется для быстрых проверок способностей без
        привязки к исходному формату данных.
        """
        if not isinstance(hero, dict):
            return set()

        raw_tokens = hero.get("hero_abilities", ())
        if isinstance(raw_tokens, str):
            token_values = raw_tokens.replace(",", " ").replace(";", " ").split()
        elif isinstance(raw_tokens, dict):
            token_values = [
                key for key, value in raw_tokens.items() if bool(value)
            ]
        elif isinstance(raw_tokens, (list, tuple, set)):
            token_values = list(raw_tokens)
        else:
            token_values = []

        return {
            str(token or "").strip().lower()
            for token in token_values
            if str(token or "").strip()
        }
    def _hero_needs_faction_hire(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить, висит ли на герое обязательство нанять фракционный юнит.

        Пока `needaunit` активен, среда может начислять штрафы за лишние ходы.
        После успешного подходящего найма флаг очищается отдельной функцией.
        """
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return False
        return self._resolve_hero_needaunit(hero) > 0
    def _pending_hire_turn_penalty(
        self,
        delta_turns: int,
        units: Optional[List[Dict]] = None,
    ) -> float:
        """Посчитать штраф за прошедшие ходы при незакрытом `needaunit`.

        Штраф применяется только если ход действительно продвинулся и герой всё
        ещё требует найма. Это shaping-сигнал: он подталкивает агента закрывать
        стартовую экономическую задачу, не меняя саму механику найма.
        """
        if delta_turns <= 0:
            return 0.0
        if not self._hero_needs_faction_hire(units=units):
            return 0.0
        return float(delta_turns) * float(self.reward_needaunit_turn_penalty)
    def _hero_has_pathfinding(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить открытие pathfinding по уровню героя."""
        return self._hero_level(units=units) >= int(self.HERO_PATHFINDING_LEVEL)
    def _hero_has_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить первый leadership-порог для расширения отряда."""
        return self._hero_level(units=units) >= int(self.HERO_LEADERSHIP_LEVEL)
    def _hero_has_second_leadership(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить второй leadership-порог для ещё одного слота найма."""
        return self._hero_level(units=units) >= int(self.HERO_SECOND_LEADERSHIP_LEVEL)
    def _hero_leadership_count(self, units: Optional[List[Dict]] = None) -> int:
        """Вернуть число открытых leadership-ступеней героя.

        Значение используется в логике лимита размера отряда: первый и второй
        пороги считаются отдельно, поэтому функция возвращает 0, 1 или 2.
        """
        level = self._hero_level(units=units)
        count = 0
        if level >= int(self.HERO_LEADERSHIP_LEVEL):
            count += 1
        if level >= int(self.HERO_SECOND_LEADERSHIP_LEVEL):
            count += 1
        return count
    def _hero_has_endurance(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить, открыт ли endurance-порог героя."""
        return self._hero_level(units=units) >= int(self.HERO_ENDURANCE_LEVEL)
    def _hero_has_strength(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить, открыт ли strength-порог героя."""
        return self._hero_level(units=units) >= int(self.HERO_STRENGTH_LEVEL)
    def _hero_has_banner_bearer(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить доступ к знамёнам через уровень или ability-token.

        Баннеры являются частью экипировки кампании, поэтому unlock может прийти
        как от уровня героя, так и напрямую из набора способностей.
        """
        if self._hero_level(units=units) >= int(self.HERO_BANNER_BEARER_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_BANNER_BEARER_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_marching_lore(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить доступ к marching lore для движения и ботинок.

        Лорд-вор получает эту специализацию сразу через `typeoflord == 3`.
        Остальные герои открывают её уровнем или явным ability-token'ом.
        """
        if int(self.typeoflord) == 3:
            return True
        if self._hero_level(units=units) >= int(self.HERO_MARCHING_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_MARCHING_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_artifact_knowledge(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить доступ к artifact knowledge для артефактов.

        Лорд-воин получает это знание как классовую замену. Для остальных
        источником служит уровень героя или явный ability-token.
        """
        if int(self.typeoflord) == 1:
            return True
        if self._hero_level(units=units) >= int(self.HERO_ARTIFACT_KNOWLEDGE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_ARTIFACT_KNOWLEDGE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_sorcery_lore(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить доступ к sorcery lore для магических предметов и посохов.

        Лорд-маг имеет эту специализацию по классу. Остальные герои получают её
        через уровень или ability-token, что сохраняет единый unlock-контракт.
        """
        if int(self.typeoflord) == 2:
            return True
        if self._hero_level(units=units) >= int(self.HERO_SORCERY_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_SORCERY_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _hero_has_book_lore(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить доступ к book lore для использования книг.

        В отличие от классовых lore, book lore открывается уровнем героя или
        явной способностью и не имеет автоматической замены от типа лорда.
        """
        if self._hero_level(units=units) >= int(self.HERO_BOOK_LORE_LEVEL):
            return True
        hero = self._resolve_travel_hero(units=units)
        return str(self.HERO_BOOK_LORE_ABILITY_KEY).lower() in self._hero_ability_tokens(hero)
    def _lord_replacement_might_level(self) -> Optional[int]:
        """Вернуть уровень, на котором классовый lore заменяет Might.

        У разных типов лорда свой профиль: воин связан с артефактами, маг с
        sorcery, вор с marching. Этот helper задаёт уровень, где такая
        специализация должна дать боевой might-бонус вместо отдельной способности.
        """
        lord_type = int(self.typeoflord)
        if lord_type == 1:
            return int(self.HERO_ARTIFACT_KNOWLEDGE_LEVEL)
        if lord_type == 2:
            return int(self.HERO_SORCERY_LORE_LEVEL)
        if lord_type == 3:
            return int(self.HERO_MARCHING_LORE_LEVEL)
        return None
    def _hero_has_might(self, units: Optional[List[Dict]] = None) -> bool:
        """Проверить, имеет ли герой Might явно или через классовую замену.

        Сначала ищется прямой ability-token Might. Если его нет, тип лорда может
        заменить Might соответствующим lore-порогом, чтобы разные классы имели
        сопоставимый боевой рост.
        """
        hero = self._resolve_travel_hero(units=units)
        if str(self.HERO_MIGHT_ABILITY_KEY).lower() in self._hero_ability_tokens(hero):
            return True
        might_level = self._lord_replacement_might_level()
        if might_level is None:
            return False
        return self._hero_level(units=units) >= int(might_level)
    def _apply_lord_replacement_might_bonus(self, hero: Optional[Dict]) -> bool:
        """Применить одноразовый damage-бонус Might от классовой замены лорда.

        Бонус записывается в `campaign_lord_might_bonus_levels`, чтобы не
        умножать урон повторно при каждом sync roster. Возвращает True только
        если урон был реально изменён в этом вызове.
        """
        if not isinstance(hero, dict) or not self._is_hero_unit(hero):
            return False

        might_level = self._lord_replacement_might_level()
        if might_level is None:
            return False

        try:
            hero_level = int(round(float(hero.get("Level", 0) or 0)))
        except (TypeError, ValueError):
            hero_level = 0
        if hero_level < int(might_level):
            return False

        applied_levels: set[int] = set()
        for level in hero.get("campaign_lord_might_bonus_levels") or []:
            try:
                applied_levels.add(int(level))
            except (TypeError, ValueError):
                continue
        if int(might_level) in applied_levels:
            return False

        base_damage = self._normalize_damage_value(
            hero.get("original_damage", hero.get("damage", 0))
        )
        boosted_damage = self._normalize_damage_value(
            float(base_damage) * float(self.HERO_MIGHT_DAMAGE_MULTIPLIER)
        )
        hero["damage"] = int(boosted_damage)
        hero["original_damage"] = int(boosted_damage)
        applied_levels.add(int(might_level))
        hero["campaign_lord_might_bonus_levels"] = sorted(applied_levels)
        return True
    @staticmethod
    def _is_empty_blue_unit(unit: Optional[Dict]) -> bool:
        """Проверить, является ли запись синего отряда пустым placeholder'ом.

        Пустые слоты могут быть представлены отсутствующим dict, именем `пусто`
        или нулевыми HP/max HP. Helper нужен, чтобы одинаково фильтровать
        реальные юниты при найме, перестановках и расчёте занятых позиций.
        """
        if not isinstance(unit, dict):
            return True
        name = str(unit.get("name", "") or "").strip().lower()
        if name == "пусто":
            return True
        hp = float(unit.get("hp", 0) or unit.get("health", 0) or 0)
        max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
        return hp <= 0 and max_hp <= 0

    def _occupied_initial_blue_positions(self, units: Optional[List[Dict]] = None) -> Tuple[int, ...]:
        """Вернуть стартовые позиции, занятые реальными синими юнитами.

        Используется для построения layout перестановок отряда. Берутся только
        позиции 7-12, потому что это боевые слоты BLUE, доступные для зелий,
        найма и swap-действий на карте.
        """
        state = units if units is not None else self._create_starting_blue_team()
        positions: List[int] = []
        for unit in state:
            if self._is_empty_blue_unit(unit):
                continue
            try:
                position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position in self.GRID_BOTTLE_POSITIONS and position not in positions:
                positions.append(position)
        return tuple(sorted(positions[:4]))

    @staticmethod
    def _build_blue_unit_swap_pairs(
        positions: Tuple[int, ...],
        extra_positions: Tuple[int, ...] = (),
    ) -> Tuple[Tuple[int, int], ...]:
        """Построить список пар позиций, которые получают swap action.

        Сначала создаются пары между занятыми стартовыми слотами. Затем
        добавляются пары с дополнительными пустыми позициями, чтобы большие
        юниты и расширенный состав могли перемещаться в свободные клетки.
        """
        pairs: List[Tuple[int, int]] = []
        normalized = tuple(int(position) for position in positions)
        for left_index, left_pos in enumerate(normalized):
            for right_pos in normalized[left_index + 1 :]:
                pairs.append((int(left_pos), int(right_pos)))
        included_positions = list(normalized)
        for extra_pos in tuple(int(position) for position in extra_positions):
            if extra_pos in included_positions:
                continue
            for other_pos in sorted(included_positions):
                pair = tuple(sorted((int(other_pos), int(extra_pos))))
                if pair not in pairs:
                    pairs.append(pair)
            included_positions.append(int(extra_pos))
        return tuple(pairs)

    def _extra_blue_unit_swap_positions(
        self,
        occupied_positions: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        """Вернуть боевые позиции BLUE, которые не заняты стартовым составом.

        Эти позиции добавляются в swap layout как потенциальные точки для
        перестановки, особенно важные для big-юнитов, которые занимают колонку
        и могут перекрывать соседний слот.
        """
        occupied = set(
            int(position)
            for position in (
                occupied_positions
                if occupied_positions is not None
                else self._occupied_initial_blue_positions(self.blue_team_state)
            )
        )
        return tuple(
            int(position)
            for position in self.GRID_BOTTLE_POSITIONS
            if int(position) not in occupied
        )

    def _reset_blue_unit_swap_state(self) -> None:
        """Пересобрать layout swap-действий и сбросить счётчики перестановок.

        Вызывается при reset или изменении состава. Функция обновляет список
        action-пар под текущий roster, очищает used-this-turn набор и общий
        счётчик успешных перестановок.
        """
        positions = self._occupied_initial_blue_positions(self.blue_team_state)
        self.GRID_SWAP_UNIT_PAIRS = self._build_blue_unit_swap_pairs(
            positions,
            self._extra_blue_unit_swap_positions(positions),
        )
        self.GRID_SWAP_UNIT_ACTION_COUNT = len(self.GRID_SWAP_UNIT_PAIRS)
        self._reset_blue_unit_swap_counters()

    def _reset_blue_unit_swap_counters(self) -> None:
        self.grid_unit_swap_actions_used_this_turn: set[int] = set()
        self.grid_unit_swaps_total = 0

    def _blue_occupancy_snapshot(
        self,
        units: Optional[List[Dict]] = None,
    ) -> Dict[str, object]:
        """Снять общий snapshot занятости BLUE-позиций для hire/swap mask."""
        state = units if units is not None else self._get_blue_state()
        direct_units_by_position: Dict[int, Dict] = {}
        big_units_by_covered_position: Dict[int, Dict] = {}
        real_direct_positions: set[int] = set()
        occupied_positions: set[int] = set()

        for unit in state or []:
            if not isinstance(unit, dict):
                continue
            try:
                unit_position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue

            is_empty = self._is_empty_blue_unit(unit)
            if unit_position not in direct_units_by_position:
                direct_units_by_position[unit_position] = unit
            if is_empty:
                continue

            real_direct_positions.add(unit_position)
            occupied_positions.add(unit_position)
            if self._is_big_blue_unit(unit):
                for covered_position in self._blue_column_positions_for_position(unit_position):
                    covered_position = int(covered_position)
                    big_units_by_covered_position.setdefault(covered_position, unit)
                    occupied_positions.add(covered_position)

        empty_by_position: Dict[int, bool] = {}
        for raw_position in self.GRID_BOTTLE_POSITIONS:
            position = int(raw_position)
            empty_by_position[position] = (
                position not in occupied_positions
                and self._is_empty_blue_unit(direct_units_by_position.get(position))
            )

        return {
            "direct_units_by_position": direct_units_by_position,
            "big_units_by_covered_position": big_units_by_covered_position,
            "real_direct_positions": real_direct_positions,
            "occupied_positions": occupied_positions,
            "empty_by_position": empty_by_position,
        }

    def _blue_unit_at_position(
        self,
        position: int,
        occupancy: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict]:
        """Найти прямого юнита BLUE, стоящего ровно в указанной позиции.

        Big-юнит, который покрывает соседний слот, здесь не подставляется:
        функция возвращает только запись с совпадающим полем `position`.
        Для покрытия big-юнитом есть отдельный helper.
        """
        if isinstance(occupancy, dict):
            direct_units = occupancy.get("direct_units_by_position", {})
            if isinstance(direct_units, dict):
                unit = direct_units.get(int(position))
                return unit if isinstance(unit, dict) else None
        for unit in self._get_blue_state():
            if not isinstance(unit, dict):
                continue
            try:
                unit_position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if unit_position == int(position):
                return unit
        return None

    @staticmethod
    def _blue_column_positions_for_position(position: int) -> Tuple[int, int]:
        """Вернуть две позиции одной вертикальной колонки BLUE.

        В боевой сетке BLUE позиции 7-9 являются передним рядом, 10-12 задним.
        Big-юнит занимает обе клетки своей колонки, поэтому многие проверки
        должны работать не с одной позицией, а с парой front/back.
        """
        position = int(position)
        if position in (7, 8, 9):
            return (position, position + 3)
        if position in (10, 11, 12):
            return (position - 3, position)
        return (position, position)

    @staticmethod
    def _blue_partner_position(position: int) -> int:
        """Вернуть вторую позицию той же колонки BLUE.

        Для переднего ряда партнёр находится сзади, для заднего - спереди.
        Если позиция не относится к стандартной BLUE-колонке, возвращается она
        сама, что делает remap безопасным для неожиданных данных.
        """
        column_positions = CampaignEconomyMixin._blue_column_positions_for_position(
            int(position)
        )
        return (
            int(column_positions[1])
            if int(column_positions[0]) == int(position)
            else int(column_positions[0])
        )

    def _is_big_blue_unit(self, unit: Optional[Dict]) -> bool:
        """Проверить, является ли запись реальным большим BLUE-юнитом.

        Placeholder'ы и пустые слоты не считаются big даже если в данных есть
        случайный флаг. Это защищает логику занятости колонок от ложных
        срабатываний на пустых позициях.
        """
        return not self._is_empty_blue_unit(unit) and bool(
            (unit or {}).get("big", False)
        )

    def _blue_big_unit_covering_position(
        self,
        position: int,
        occupancy: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict]:
        """Найти big-юнита, который покрывает указанную позицию колонки.

        Big-юнит может физически храниться в передней позиции, но занимать и
        заднюю. Helper нужен для swap, найма и occupancy-проверок, где важна
        фактическая занятость клетки, а не только поле `position`.
        """
        position = int(position)
        if isinstance(occupancy, dict):
            covered_units = occupancy.get("big_units_by_covered_position", {})
            if isinstance(covered_units, dict):
                unit = covered_units.get(position)
                return unit if isinstance(unit, dict) else None
        for unit in self._get_blue_state():
            if not self._is_big_blue_unit(unit):
                continue
            try:
                unit_position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position in self._blue_column_positions_for_position(unit_position):
                return unit
        return None

    def _blue_unit_swap_position_snapshot(
        self,
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict], set[int]]:
        """Снять быстрый снимок занятости позиций для расчёта swap mask.

        Возвращает прямые юниты по позиции, big-юнитов по покрываемым клеткам и
        множество позиций с реальными прямыми юнитами. Это позволяет собрать
        маску всех swap actions без повторного сканирования roster для каждой
        пары.
        """
        snapshot = self._blue_occupancy_snapshot()
        return (
            snapshot["direct_units_by_position"],
            snapshot["big_units_by_covered_position"],
            snapshot["real_direct_positions"],
        )

    def _blue_swap_endpoint_from_snapshot(
        self,
        selected_position: int,
        direct_units_by_position: Dict[int, Dict],
        big_units_by_covered_position: Dict[int, Dict],
    ) -> Dict[str, object]:
        """Собрать описание swap-края из заранее снятого snapshot.

        Endpoint хранит выбранную позицию, фактический unit, исходную позицию
        unit'а, признак big и пару клеток его колонки. Такой dict используется
        и для mask, и для выполнения swap без повторной нормализации.
        """
        selected_position = int(selected_position)
        unit = direct_units_by_position.get(selected_position)
        if self._is_empty_blue_unit(unit):
            unit = big_units_by_covered_position.get(selected_position)
        is_big = self._is_big_blue_unit(unit)
        unit_position = None
        if unit is not None and not self._is_empty_blue_unit(unit):
            try:
                unit_position = int(unit.get("position", selected_position) or selected_position)
            except (TypeError, ValueError):
                unit_position = selected_position
        return {
            "selected_position": selected_position,
            "unit": unit,
            "unit_position": unit_position,
            "is_big": bool(is_big),
            "column_positions": self._blue_column_positions_for_position(selected_position),
        }

    def _blue_swap_endpoint(self, selected_position: int) -> Dict[str, object]:
        """Собрать описание swap-края напрямую из текущего roster.

        Это runtime-версия endpoint builder для выполнения конкретного action.
        Если выбранная клетка пуста, но покрыта big-юнитом, endpoint будет
        ссылаться на этого big-юнита.
        """
        selected_position = int(selected_position)
        unit = self._blue_unit_at_position(selected_position)
        if self._is_empty_blue_unit(unit):
            covering_big = self._blue_big_unit_covering_position(selected_position)
            if covering_big is not None:
                unit = covering_big
        is_big = self._is_big_blue_unit(unit)
        unit_position = None
        if unit is not None and not self._is_empty_blue_unit(unit):
            try:
                unit_position = int(unit.get("position", selected_position) or selected_position)
            except (TypeError, ValueError):
                unit_position = selected_position
        return {
            "selected_position": selected_position,
            "unit": unit,
            "unit_position": unit_position,
            "is_big": bool(is_big),
            "column_positions": self._blue_column_positions_for_position(selected_position),
        }

    def _can_swap_blue_unit_positions(self, action_offset: int) -> bool:
        """Проверить допустимость конкретного swap action.

        Запрещаются уже использованные в текущем ходу действия, индексы вне
        layout, swap одного и того же big-юнита с самим собой и обмен пустых
        обычных слотов. Для big-юнитов разрешён перенос между разными колонками.
        """
        action_offset = int(action_offset)
        if action_offset in getattr(self, "grid_unit_swap_actions_used_this_turn", set()):
            return False
        pairs = tuple(getattr(self, "GRID_SWAP_UNIT_PAIRS", ()) or ())
        if action_offset < 0 or action_offset >= len(pairs):
            return False
        pos_a, pos_b = pairs[action_offset]
        endpoint_a = self._blue_swap_endpoint(int(pos_a))
        endpoint_b = self._blue_swap_endpoint(int(pos_b))
        unit_a = endpoint_a["unit"]
        unit_b = endpoint_b["unit"]
        if unit_a is not None and unit_a is unit_b:
            return False
        if bool(endpoint_a["is_big"]) or bool(endpoint_b["is_big"]):
            return tuple(endpoint_a["column_positions"]) != tuple(
                endpoint_b["column_positions"]
            )
        return not self._is_empty_blue_unit(unit_a) or not self._is_empty_blue_unit(unit_b)

    def _blue_unit_swap_action_mask_values(self) -> Tuple[bool, ...]:
        """Построить bool-маску для всех действий перестановки отряда.

        В отличие от `_can_swap_blue_unit_positions`, функция работает batch'ем:
        один раз снимает snapshot roster и затем делает дешёвые lookup'и для
        каждой пары. Это важно, потому что action mask вызывается очень часто.
        """
        pairs = tuple(getattr(self, "GRID_SWAP_UNIT_PAIRS", ()) or ())
        if not pairs:
            return ()

        used_actions = getattr(self, "grid_unit_swap_actions_used_this_turn", set())
        (
            direct_units_by_position,
            big_units_by_covered_position,
            real_direct_positions,
        ) = self._blue_unit_swap_position_snapshot()
        endpoint_by_position: Dict[int, Dict[str, object]] = {}
        for pos_a, pos_b in pairs:
            for position in (int(pos_a), int(pos_b)):
                if position not in endpoint_by_position:
                    endpoint_by_position[position] = self._blue_swap_endpoint_from_snapshot(
                        position,
                        direct_units_by_position,
                        big_units_by_covered_position,
                    )
        values: List[bool] = []

        for action_offset, pair in enumerate(pairs):
            if action_offset in used_actions:
                values.append(False)
                continue

            pos_a, pos_b = pair
            endpoint_a = endpoint_by_position[int(pos_a)]
            endpoint_b = endpoint_by_position[int(pos_b)]
            unit_a = endpoint_a["unit"]
            unit_b = endpoint_b["unit"]
            if unit_a is not None and unit_a is unit_b:
                values.append(False)
                continue
            if bool(endpoint_a["is_big"]) or bool(endpoint_b["is_big"]):
                values.append(
                    tuple(endpoint_a["column_positions"])
                    != tuple(endpoint_b["column_positions"])
                )
                continue

            values.append(
                int(pos_a) in real_direct_positions
                or int(pos_b) in real_direct_positions
            )

        return tuple(values)

    @staticmethod
    def _swap_position_membership(position_set: set[int], pos_a: int, pos_b: int) -> None:
        """Поменять членство двух позиций внутри set активного эффекта.

        Если эффект висел на `pos_a`, он переносится на `pos_b`, и наоборот.
        Если обе позиции либо обе есть, либо обе отсутствуют, set не меняется:
        это сохраняет корректность для AoE/дублирующих состояний.
        """
        has_a = int(pos_a) in position_set
        has_b = int(pos_b) in position_set
        if has_a == has_b:
            return
        if has_a:
            position_set.discard(int(pos_a))
            position_set.add(int(pos_b))
        else:
            position_set.discard(int(pos_b))
            position_set.add(int(pos_a))

    def _swap_blue_unit_position_effects(self, pos_a: int, pos_b: int) -> None:
        """Перенести все позиционные potion/ward эффекты при обычном swap.

        Эффекты зелий хранятся как sets позиций, а не как ссылки на юнитов.
        После обмена двух юнитов эти sets нужно синхронно поменять, иначе buff
        останется на старой клетке и начнёт применяться к другому юниту.
        """
        effect_sets = (
            self.active_invulnerability_potion_positions,
            self.active_strength_potion_positions,
            self.active_energy_elixir_positions,
            self.active_haste_elixir_positions,
            self.active_fire_ward_positions,
            self.active_earth_ward_positions,
            self.active_water_ward_positions,
            self.active_air_ward_positions,
        )
        for position_set in effect_sets:
            self._swap_position_membership(position_set, int(pos_a), int(pos_b))
        for position_set in getattr(self, "active_potion_effect_positions", {}).values():
            self._swap_position_membership(position_set, int(pos_a), int(pos_b))

    @staticmethod
    def _remap_position_membership(position_set: set[int], position_map: Dict[int, int]) -> None:
        """Переотобразить set позиций через произвольную карту перемещений.

        Используется для сложных перестановок big-area, где двигается не одна
        пара клеток, а несколько участников колонки. Позиции без записи в map
        остаются на месте.
        """
        if not position_set:
            return
        remapped = {int(position_map.get(int(position), int(position))) for position in position_set}
        position_set.clear()
        position_set.update(remapped)

    def _remap_blue_unit_position_effects(self, position_map: Dict[int, int]) -> None:
        """Применить произвольный remap ко всем активным позиционным эффектам.

        Big-юнит может переносить сразу две покрываемые клетки, поэтому простой
        swap `a <-> b` недостаточен. Эта функция обновляет все known sets зелий,
        ward'ов и динамических potion effects по единой карте перемещения.
        """
        effect_sets = (
            self.active_invulnerability_potion_positions,
            self.active_strength_potion_positions,
            self.active_energy_elixir_positions,
            self.active_haste_elixir_positions,
            self.active_fire_ward_positions,
            self.active_earth_ward_positions,
            self.active_water_ward_positions,
            self.active_air_ward_positions,
        )
        for position_set in effect_sets:
            self._remap_position_membership(position_set, position_map)
        for position_set in getattr(self, "active_potion_effect_positions", {}).values():
            self._remap_position_membership(position_set, position_map)

    @staticmethod
    def _blue_stand_for_position(position: int) -> str:
        """Вернуть стойку `ahead/behind` для позиции BLUE."""
        return "ahead" if int(position) in (7, 8, 9) else "behind"

    def _blue_area_occupants(self, positions: Tuple[int, int]) -> List[Dict]:
        """Вернуть фактических юнитов, занимающих указанную колонку BLUE.

        Если колонку покрывает big-юнит, он считается единственным occupant'ом.
        Иначе возвращаются обычные юниты в передней/задней клетке. Это нужно
        для корректной перестановки целых big-area блоков.
        """
        occupancy = self._blue_occupancy_snapshot()
        big_units: List[Dict] = []
        for position in positions:
            big_unit = self._blue_big_unit_covering_position(
                int(position),
                occupancy=occupancy,
            )
            if big_unit is not None and not any(big_unit is unit for unit in big_units):
                big_units.append(big_unit)
        if big_units:
            return big_units[:1]

        occupants: List[Dict] = []
        for position in positions:
            unit = self._blue_unit_at_position(int(position), occupancy=occupancy)
            if not self._is_empty_blue_unit(unit):
                occupants.append(unit)
        return occupants

    @staticmethod
    def _corresponding_blue_area_position(
        old_position: int,
        from_positions: Tuple[int, int],
        to_positions: Tuple[int, int],
    ) -> int:
        """Найти соответствующую клетку в другой колонке BLUE.

        Обычный юнит из передней клетки остаётся в передней клетке новой
        колонки, а юнит из задней - в задней. Это сохраняет stand при переносе
        area между колонками.
        """
        return int(to_positions[1]) if int(old_position) == int(from_positions[1]) else int(to_positions[0])

    def _unit_swap_reward(self) -> float:
        """Вернуть reward-штраф за успешную перестановку юнитов.

        Штраф берётся из reward-config и всегда неположительный. Он нужен, чтобы
        агент не спамил бесплатные swap-действия без стратегической причины.
        """
        return -max(0.0, float(getattr(self, "reward_unit_swap_penalty", 0.0) or 0.0))

    def _step_swap_big_blue_unit_positions(
        self,
        action_offset: int,
        pos_a: int,
        pos_b: int,
        endpoint_a: Dict[str, object],
        endpoint_b: Dict[str, object],
        base_info: Dict[str, object],
    ):
        """Выполнить перестановку, где хотя бы один endpoint связан с big-юнитом.

        Big-юнит занимает колонку, поэтому действие меняет не две отдельные
        клетки, а две area-колонки. Функция пересобирает affected positions,
        переносит обычных и больших юнитов, remap'ит активные эффекты и
        возвращает стандартный grid-step result с подробным info.
        """
        area_a = tuple(int(pos) for pos in endpoint_a["column_positions"])
        area_b = tuple(int(pos) for pos in endpoint_b["column_positions"])
        if area_a == area_b:
            grid_obs = self._get_grid_obs()
            info = dict(base_info)
            info.update(
                {
                    "unit_swap_applied": False,
                    "unit_swap_positions": (int(pos_a), int(pos_b)),
                    "unit_swap_action_offset": int(action_offset),
                    "unit_swap_invalid_reason": "same_big_unit_area",
                }
            )
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        occupants_a = self._blue_area_occupants(area_a)
        occupants_b = self._blue_area_occupants(area_b)
        selected_a = int(endpoint_a["selected_position"])
        selected_b = int(endpoint_b["selected_position"])

        moves: List[Tuple[Dict, int, int]] = []
        for unit in occupants_a:
            old_position = int(unit.get("position", selected_a) or selected_a)
            if self._is_big_blue_unit(unit):
                new_position = int(area_b[0])
            else:
                new_position = self._corresponding_blue_area_position(
                    old_position,
                    area_a,
                    area_b,
                )
            moves.append((unit, old_position, int(new_position)))
        for unit in occupants_b:
            old_position = int(unit.get("position", selected_b) or selected_b)
            if self._is_big_blue_unit(unit):
                new_position = int(area_a[0])
            else:
                new_position = self._corresponding_blue_area_position(
                    old_position,
                    area_b,
                    area_a,
                )
            moves.append((unit, old_position, int(new_position)))

        affected_positions = set(area_a) | set(area_b)
        units_by_position: Dict[int, Dict] = {
            int(position): placeholder_unit("blue", int(position))
            for position in affected_positions
        }
        position_map = {int(position): int(position) for position in affected_positions}
        moved_unit_ids: set[int] = set()
        move_infos: List[Dict[str, object]] = []
        for unit, old_position, new_position in moves:
            moved_unit_ids.add(id(unit))
            unit["position"] = int(new_position)
            unit["stand"] = self._blue_stand_for_position(int(new_position))
            units_by_position[int(new_position)] = unit
            position_map[int(old_position)] = int(new_position)
            old_partner = self._blue_partner_position(int(old_position))
            new_partner = self._blue_partner_position(int(new_position))
            if int(old_partner) in affected_positions:
                position_map[int(old_partner)] = int(new_partner)
            move_infos.append(
                {
                    "name": str(unit.get("name", "") or ""),
                    "from": int(old_position),
                    "to": int(new_position),
                    "big": bool(unit.get("big", False)),
                }
            )

        rebuilt_state: List[Dict] = []
        for unit in self._get_blue_state():
            if not isinstance(unit, dict):
                continue
            try:
                position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                position = -1
            if id(unit) in moved_unit_ids:
                continue
            if position in affected_positions:
                continue
            rebuilt_state.append(unit)
        for position in sorted(affected_positions):
            rebuilt_state.append(units_by_position[int(position)])
        rebuilt_state.sort(key=lambda unit: int(unit.get("position", 999) or 999))
        self.blue_team_state = rebuilt_state
        self._mark_equipment_dirty()
        self._remap_blue_unit_position_effects(position_map)
        self._sync_hero_progression_flags(self.blue_team_state)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
        self.grid_unit_swap_actions_used_this_turn.add(int(action_offset))
        self.grid_unit_swaps_total += 1

        move_text = ", ".join(
            f"{move_info['name']} pos{move_info['from']}->pos{move_info['to']}"
            for move_info in move_infos
        )
        self._log(f"Перестановка BLUE big-area: {move_text}")
        grid_obs = self._get_grid_obs()
        info = dict(base_info)
        swap_reward = self._unit_swap_reward()
        info.update(
            {
                "unit_swap_applied": True,
                "unit_swap_big_area": True,
                "unit_swap_positions": (int(pos_a), int(pos_b)),
                "unit_swap_action_offset": int(action_offset),
                "unit_swap_affected_positions": tuple(sorted(affected_positions)),
                "unit_swap_unit_moves": move_infos,
                "unit_swap_penalty": abs(float(swap_reward)),
                "unit_swap_reward": float(swap_reward),
                "unit_swaps_total": int(self.grid_unit_swaps_total),
            }
        )
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=swap_reward,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_swap_blue_unit_positions(self, action: int):
        """Обработать grid action перестановки двух позиций синего отряда.

        Функция переводит глобальный action в offset layout'а, проверяет mask,
        отдельно делегирует big-area swap, а для обычных юнитов меняет position,
        stand и связанные potion/ward эффекты. В результате возвращается обычный
        tuple step с reward-штрафом и диагностическим info для callback'ов.
        """
        action_offset = int(action) - int(self.GRID_SWAP_UNIT_ACTION_START)
        pairs = tuple(getattr(self, "GRID_SWAP_UNIT_PAIRS", ()) or ())
        base_info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "unit_swap_action": True,
        }
        if action_offset < 0 or action_offset >= len(pairs):
            grid_obs = self._get_grid_obs()
            info = dict(base_info)
            info.update(
                {
                    "unit_swap_applied": False,
                    "unit_swap_invalid_reason": "action_out_of_range",
                }
            )
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        pos_a, pos_b = pairs[action_offset]
        endpoint_a = self._blue_swap_endpoint(int(pos_a))
        endpoint_b = self._blue_swap_endpoint(int(pos_b))
        if not self._can_swap_blue_unit_positions(action_offset):
            grid_obs = self._get_grid_obs()
            info = dict(base_info)
            info.update(
                {
                    "unit_swap_applied": False,
                    "unit_swap_positions": (int(pos_a), int(pos_b)),
                    "unit_swap_action_offset": int(action_offset),
                    "unit_swap_invalid_reason": "masked_or_missing_unit",
                }
            )
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        unit_a = self._blue_unit_at_position(int(pos_a))
        unit_b = self._blue_unit_at_position(int(pos_b))
        if bool(endpoint_a["is_big"]) or bool(endpoint_b["is_big"]):
            return self._step_swap_big_blue_unit_positions(
                action_offset,
                int(pos_a),
                int(pos_b),
                endpoint_a,
                endpoint_b,
                base_info,
            )
        real_a = not self._is_empty_blue_unit(unit_a)
        real_b = not self._is_empty_blue_unit(unit_b)
        if not real_a and not real_b:
            grid_obs = self._get_grid_obs()
            info = dict(base_info)
            info.update(
                {
                    "unit_swap_applied": False,
                    "unit_swap_positions": (int(pos_a), int(pos_b)),
                    "unit_swap_action_offset": int(action_offset),
                    "unit_swap_invalid_reason": "missing_unit",
                }
            )
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        unit_swap_move_to_empty = False
        unit_swap_unit_names: Tuple[str, ...]
        if real_a and real_b:
            name_a = str(unit_a.get("name", f"pos{pos_a}") or f"pos{pos_a}")
            name_b = str(unit_b.get("name", f"pos{pos_b}") or f"pos{pos_b}")
            unit_a["position"] = int(pos_b)
            unit_b["position"] = int(pos_a)
            unit_a["stand"] = self._blue_stand_for_position(int(pos_b))
            unit_b["stand"] = self._blue_stand_for_position(int(pos_a))
            self._swap_blue_unit_position_effects(int(pos_a), int(pos_b))
            unit_swap_unit_names = (name_a, name_b)
            log_message = (
                f"Перестановка BLUE: {name_a} pos{pos_a}->pos{pos_b}, "
                f"{name_b} pos{pos_b}->pos{pos_a}"
            )
        else:
            moving_unit = unit_a if real_a else unit_b
            old_pos = int(pos_a if real_a else pos_b)
            new_pos = int(pos_b if real_a else pos_a)
            moving_name = str(
                moving_unit.get("name", f"pos{old_pos}") or f"pos{old_pos}"
            )
            moving_unit["position"] = int(new_pos)
            moving_unit["stand"] = self._blue_stand_for_position(int(new_pos))

            rebuilt_state: List[Dict] = []
            moving_unit_id = id(moving_unit)
            old_placeholder_added = False
            for unit in self._get_blue_state():
                if not isinstance(unit, dict):
                    continue
                try:
                    unit_position = int(unit.get("position", -1) or -1)
                except (TypeError, ValueError):
                    unit_position = -1
                if id(unit) == moving_unit_id:
                    if not old_placeholder_added:
                        rebuilt_state.append(placeholder_unit("blue", int(old_pos)))
                        old_placeholder_added = True
                    continue
                if unit_position == int(new_pos) and self._is_empty_blue_unit(unit):
                    continue
                rebuilt_state.append(unit)
            if not old_placeholder_added:
                rebuilt_state.append(placeholder_unit("blue", int(old_pos)))
            rebuilt_state.append(moving_unit)
            self.blue_team_state = rebuilt_state
            self._swap_blue_unit_position_effects(int(old_pos), int(new_pos))
            unit_swap_move_to_empty = True
            unit_swap_unit_names = (moving_name,)
            log_message = f"Перемещение BLUE: {moving_name} pos{old_pos}->pos{new_pos}"

        self.blue_team_state.sort(key=lambda unit: int(unit.get("position", 999) or 999))
        self._sync_hero_progression_flags(self.blue_team_state)
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state)
        self.grid_unit_swap_actions_used_this_turn.add(int(action_offset))
        self.grid_unit_swaps_total += 1

        self._log(log_message)
        grid_obs = self._get_grid_obs()
        info = dict(base_info)
        swap_reward = self._unit_swap_reward()
        info.update(
            {
                "unit_swap_applied": True,
                "unit_swap_move_to_empty": bool(unit_swap_move_to_empty),
                "unit_swap_positions": (int(pos_a), int(pos_b)),
                "unit_swap_action_offset": int(action_offset),
                "unit_swap_unit_names": unit_swap_unit_names,
                "unit_swap_penalty": abs(float(swap_reward)),
                "unit_swap_reward": float(swap_reward),
                "unit_swaps_total": int(self.grid_unit_swaps_total),
            }
        )
        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=swap_reward,
            terminated=False,
            truncated=False,
            info=info,
        )
    def _hire_option(self, action_idx: int) -> Optional[Dict[str, object]]:
        """Вернуть копию фракционной hire-option по индексу action layout.

        Копия защищает `active_hire_options` от случайной мутации во время
        проверки или выполнения найма. Некорректный индекс означает отсутствие
        доступного варианта и возвращает None.
        """
        if 0 <= int(action_idx) < len(self.active_hire_options):
            return dict(self.active_hire_options[int(action_idx)])
        return None
    def _hire_positions_for_role(self, role: object) -> Tuple[int, ...]:
        """Сопоставить роль фракционного юнита с допустимыми позициями найма.

        Warriors нанимаются в передний ряд, а mage/archer/support - в задний.
        Неизвестная роль возвращает пустой tuple, чтобы mask не разрешала
        действие с некорректными данными.
        """
        role_name = str(role or "").strip().lower()
        if role_name == "warrior":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if role_name in {"mage", "archer", "support"}:
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()
    def _hire_positions_for_stand(self, stand: object) -> Tuple[int, ...]:
        """Сопоставить стойку наёмника с позициями BLUE.

        Mercenary roster хранит не роль, а stand: `ahead` для переднего ряда и
        `behind` для заднего. Пустой результат блокирует найм через mask.
        """
        stand_name = str(stand or "").strip().lower()
        if stand_name == "ahead":
            return tuple(int(pos) for pos in self.HIRE_FRONT_POSITIONS)
        if stand_name == "behind":
            return tuple(int(pos) for pos in self.HIRE_BACK_POSITIONS)
        return tuple()
    def _is_big_hire_option(self, option: Optional[Dict[str, object]]) -> bool:
        """Проверить, описывает ли hire-option специальный big-unit найм.

        Big-юниты занимают целую колонку и выполняются отдельным путём найма,
        поэтому обычный `role`/позиционный алгоритм для них не подходит.
        """
        if not isinstance(option, dict):
            return False
        return (
            str(option.get("hire_kind", "") or "").strip().lower()
            == str(self.HIRE_BIG_UNIT_KIND).lower()
        )
    def _faction_big_hire_option_for_capital(
        self,
        capital: Optional[int],
    ) -> Optional[Dict[str, object]]:
        """Создать synthetic hire-option для большого юнита текущей фракции.

        Данные обычного HIRE_D2 не всегда содержат отдельную запись для
        leadership big-unit. Helper добавляет её динамически по Realcapital,
        чтобы action layout имел единый список hire options.
        """
        try:
            capital_value = self._normalize_capital_id(int(capital))
        except (TypeError, ValueError):
            capital_value = self._normalize_capital_id(getattr(self, "Realcapital", 1))

        unit_name = str(self.HIRE_BIG_UNIT_NAME_BY_CAPITAL.get(int(capital_value), "") or "").strip()
        if not unit_name:
            return None
        return {
            "id": f"big_d2_h{int(capital_value):03d}",
            "name": unit_name,
            "role": "big",
            "gold": float(self.HIRE_BIG_UNIT_GOLD_COST),
            "hire_kind": str(self.HIRE_BIG_UNIT_KIND),
        }
    def _is_blue_position_empty_for_hire(
        self,
        position: int,
        occupancy: Optional[Dict[str, object]] = None,
    ) -> bool:
        """Проверить, свободна ли позиция BLUE для найма.

        Позиция считается занятой не только прямым юнитом, но и big-юнитом,
        который покрывает эту клетку своей колонкой.
        """
        if occupancy is None:
            occupancy = self._blue_occupancy_snapshot()
        empty_by_position = occupancy.get("empty_by_position", {})
        if isinstance(empty_by_position, dict) and int(position) in empty_by_position:
            return bool(empty_by_position[int(position)])
        if self._blue_big_unit_covering_position(int(position), occupancy=occupancy) is not None:
            return False
        return self._is_empty_blue_unit(
            self._blue_unit_at_position(int(position), occupancy=occupancy)
        )
    def _find_first_empty_blue_position(
        self,
        positions: Tuple[int, ...],
        occupancy: Optional[Dict[str, object]] = None,
    ) -> Optional[int]:
        """Найти первую свободную позицию из заданного списка кандидатов."""
        if occupancy is None:
            occupancy = self._blue_occupancy_snapshot()
        for position in positions:
            if self._is_blue_position_empty_for_hire(int(position), occupancy=occupancy):
                return int(position)
        return None
    def _find_first_empty_blue_column_front_position(
        self,
        occupancy: Optional[Dict[str, object]] = None,
    ) -> Optional[int]:
        """Найти переднюю клетку первой полностью свободной BLUE-колонки.

        Используется для найма big-юнита: ему нужна не одна клетка, а вся
        вертикальная колонка front/back.
        """
        if occupancy is None:
            occupancy = self._blue_occupancy_snapshot()
        for front_position in self.HIRE_BIG_FRONT_POSITIONS:
            column_positions = self._blue_column_positions_for_position(int(front_position))
            if all(
                self._is_blue_position_empty_for_hire(
                    int(position),
                    occupancy=occupancy,
                )
                for position in column_positions
            ):
                return int(column_positions[0])
        return None
    def _blue_real_unit_count(self, units: Optional[List[Dict]] = None) -> int:
        """Посчитать реальные юниты BLUE, игнорируя placeholder'ы.

        Big-юнит учитывается один раз даже если в roster есть дополнительная
        placeholder-запись партнёрской клетки. Это число используется в логике
        leadership hire и проверке, нанимал ли герой уже дополнительного юнита.
        """
        roster = units if units is not None else self._get_blue_state()
        count = 0
        seen_big_units: set[int] = set()
        for unit in roster or []:
            if self._is_empty_blue_unit(unit):
                continue
            if self._is_big_blue_unit(unit):
                unit_id = id(unit)
                if unit_id in seen_big_units:
                    continue
                seen_big_units.add(unit_id)
            count += 1
        return int(count)
    def _starting_blue_real_unit_count(self) -> int:
        """Вернуть число реальных юнитов в стартовом составе BLUE."""
        return int(len(self._occupied_initial_blue_positions(self._create_starting_blue_team())))
    def _hero_has_completed_leadership_hire(self, hero: Optional[Dict]) -> bool:
        """Проверить, закрыл ли герой первый leadership-найм.

        Современный путь читает явный флаг `campaign_hired_after_first_leadership`.
        Fallback сравнивает текущий размер отряда со стартовым, чтобы старые
        сохранения и тестовые состояния оставались совместимыми.
        """
        if not isinstance(hero, dict):
            return False
        if "campaign_hired_after_first_leadership" in hero:
            return self._truthy_unit_flag(hero, "campaign_hired_after_first_leadership")
        return self._blue_real_unit_count() > self._starting_blue_real_unit_count()
    def _hero_can_use_big_leadership_hire(self, hero: Optional[Dict]) -> bool:
        """Проверить, может ли герой нанять фракционного big-юнита.

        Требуется второй leadership-порог, активный `needaunit` и отсутствие уже
        выполненного первого leadership-найма. Так big-unit найм не становится
        бесплатным расширением сверх сценарного лимита.
        """
        if not isinstance(hero, dict):
            return False
        if not self._hero_has_second_leadership(units=[hero]):
            return False
        if self._resolve_hero_needaunit(hero) <= 0:
            return False
        return not self._hero_has_completed_leadership_hire(hero)
    def _clear_hero_needaunit_after_hire(self, units: List[Dict]) -> None:
        """Снять `needaunit` с героя после успешного найма.

        Если герой уже имеет leadership, дополнительно ставится флаг, что первый
        leadership-найм был закрыт. Это влияет на доступность big-unit найма.
        """
        hero = self._resolve_travel_hero(units=units)
        if hero is None:
            return
        hero["needaunit"] = 0
        if self._hero_has_leadership(units=[hero]):
            hero["campaign_hired_after_first_leadership"] = 1
    def _can_hire_faction_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        """Проверить доступность обычного фракционного найма из столицы.

        Проверяются: корректность option, нахождение в замке, наличие героя,
        leadership, активный `needaunit`, золото, существование unit data и
        свободная позиция подходящего ряда.
        """
        if not isinstance(option, dict):
            return False
        if self._is_big_hire_option(option):
            return self._can_hire_faction_big_unit_option(option)
        if not self._is_at_castle():
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False
        if self._resolve_hero_needaunit(hero) <= 0:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        if not unit_name or not positions:
            return False
        if self._find_unit_data_by_name(unit_name) is None:
            return False
        return self._find_first_empty_blue_position(positions) is not None
    def _can_hire_faction_big_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        """Проверить доступность фракционного big-unit найма.

        Это более строгий вариант hire-check: помимо замка, золота и unit data
        требуется право героя на big leadership hire и полностью свободная
        колонка BLUE.
        """
        if not isinstance(option, dict):
            return False
        if not self._is_at_castle():
            return False

        hero = self._resolve_travel_hero()
        if not self._hero_can_use_big_leadership_hire(hero):
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        unit_name = str(option.get("name", "") or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or not self._unit_data_is_big(unit_data):
            return False
        return self._find_first_empty_blue_column_front_position() is not None
    def _faction_hire_action_mask_values(self) -> Tuple[bool, ...]:
        """Batch-доступность всех фракционных hire actions для action mask."""
        options = tuple(getattr(self, "active_hire_options", ()) or ())
        if not options:
            return ()
        if not self._is_at_castle():
            return tuple(False for _ in options)

        hero = self._resolve_travel_hero()
        if hero is None:
            return tuple(False for _ in options)

        occupancy = self._blue_occupancy_snapshot()
        gold_available = float(self.gold or 0.0)
        has_leadership = self._hero_has_leadership(units=[hero])
        needs_hire = self._resolve_hero_needaunit(hero) > 0
        can_use_big_hire = self._hero_can_use_big_leadership_hire(hero)
        first_empty_by_positions: Dict[Tuple[int, ...], Optional[int]] = {}
        first_empty_big_column = self._find_first_empty_blue_column_front_position(
            occupancy=occupancy,
        )

        values: List[bool] = []
        for option in options:
            if not isinstance(option, dict):
                values.append(False)
                continue

            try:
                cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                values.append(False)
                continue
            if gold_available < cost:
                values.append(False)
                continue

            if self._is_big_hire_option(option):
                if not can_use_big_hire or first_empty_big_column is None:
                    values.append(False)
                    continue
                unit_name = str(option.get("name", "") or "").strip()
                unit_data = self._find_unit_data_by_name(unit_name)
                values.append(
                    bool(unit_name)
                    and unit_data is not None
                    and self._unit_data_is_big(unit_data)
                )
                continue

            if not has_leadership or not needs_hire:
                values.append(False)
                continue
            unit_name = str(option.get("name", "") or "").strip()
            positions = self._hire_positions_for_role(option.get("role"))
            if not unit_name or not positions:
                values.append(False)
                continue
            unit_data = self._find_unit_data_by_name(unit_name)
            if unit_data is None:
                values.append(False)
                continue
            if positions not in first_empty_by_positions:
                first_empty_by_positions[positions] = self._find_first_empty_blue_position(
                    positions,
                    occupancy=occupancy,
                )
            values.append(first_empty_by_positions[positions] is not None)
        return tuple(values)
    def _hire_faction_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float]:
        """Выполнить обычный фракционный найм и списать золото.

        Успешный найм создаёт unit из справочника, ставит его в первую свободную
        позицию подходящего ряда, очищает `needaunit`, помечает экипировку dirty
        и возвращает результат для step-info.
        """
        if self._is_big_hire_option(option):
            return self._hire_faction_big_unit_option(option)
        if not self._can_hire_faction_unit_option(option):
            return False, None, None, 0.0

        assert isinstance(option, dict)
        unit_name = str(option.get("name", "") or "").strip()
        positions = self._hire_positions_for_role(option.get("role"))
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0

        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None:
            return False, None, None, 0.0

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        self._clear_hero_needaunit_after_hire(state)

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        self._mark_equipment_dirty()
        self._sync_hero_progression_flags(state)
        self._log(
            f'Найм в городе: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f})"
        )
        return True, unit_name, int(target_pos), cost
    def _hire_faction_big_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float]:
        """Выполнить найм большого фракционного юнита.

        Big-юнит ставится в переднюю клетку свободной колонки, а задняя клетка
        заполняется placeholder'ом. Это сохраняет представление roster
        совместимым с боевой сеткой, где big-юнит занимает две позиции.
        """
        if not self._can_hire_faction_big_unit_option(option):
            return False, None, None, 0.0

        assert isinstance(option, dict)
        unit_name = str(option.get("name", "") or "").strip()
        target_pos = self._find_first_empty_blue_column_front_position()
        if target_pos is None:
            return False, None, None, 0.0
        partner_pos = self._blue_partner_position(int(target_pos))

        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or not self._unit_data_is_big(unit_data):
            return False, None, None, 0.0

        recruited_unit = self._build_unit_from_data(unit_data, "blue", int(target_pos))
        recruited_unit["position"] = int(target_pos)
        recruited_unit["stand"] = "ahead"
        recruited_unit["big"] = True

        state = self._get_blue_state()
        replaced_front = False
        replaced_back = False
        for idx, unit in enumerate(state):
            try:
                position = int(unit.get("position", -1) or -1)
            except (TypeError, ValueError):
                continue
            if position == int(target_pos):
                state[idx] = deepcopy(recruited_unit)
                replaced_front = True
            elif position == int(partner_pos):
                state[idx] = placeholder_unit("blue", int(partner_pos))
                replaced_back = True
        if not replaced_front:
            state.append(deepcopy(recruited_unit))
        if not replaced_back:
            state.append(placeholder_unit("blue", int(partner_pos)))
        state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        self._clear_hero_needaunit_after_hire(state)

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        self._mark_equipment_dirty()
        self._sync_hero_progression_flags(state)
        self._log(
            f'Big hire in city: "{unit_name}" hired at position {int(target_pos)} '
            f"(column={int(target_pos)}/{int(partner_pos)}, cost={cost:.1f} gold, "
            f"gold_left={float(self.gold):.1f})"
        )
        return True, unit_name, int(target_pos), cost
    @staticmethod
    def _unit_data_is_big(unit_data: Optional[Dict]) -> bool:
        """Проверить флаг большого юнита в unit data.

        Поддерживаются канонический ключ `big` и legacy-поле размера из данных.
        Невалидные числовые значения приводятся к bool исходного значения.
        """
        if not isinstance(unit_data, dict):
            return False
        raw_value = unit_data.get("размер", unit_data.get("big", 0))
        try:
            return bool(int(raw_value or 0))
        except (TypeError, ValueError):
            return bool(raw_value)
    def _can_hire_mercenary_unit_option(self, option: Optional[Dict[str, object]]) -> bool:
        """Проверить доступность найма конкретного наёмника на текущей клетке.

        Для наёмников проверяются site на позиции героя, stock, leadership,
        наличие unit data, запрет big-юнитов, подходящий stand, золото и
        свободная позиция в нужном ряду.
        """
        if not isinstance(option, dict):
            return False

        site_name = str(option.get("site_name", "") or "")
        if not site_name or site_name not in self._mercenary_sites_at_position(self.grid_env.agent_pos):
            return False
        if max(0, int(option.get("stock", 0) or 0)) <= 0:
            return False

        hero = self._resolve_travel_hero()
        if hero is None:
            return False
        if not self._hero_has_leadership(units=[hero]):
            return False

        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        if not unit_name or not positions:
            return False

        try:
            cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        if float(self.gold or 0.0) < cost:
            return False

        return self._find_first_empty_blue_position(positions) is not None
    def _mercenary_hire_action_mask_values(self) -> Tuple[bool, ...]:
        """Batch-доступность всех действий найма наёмников."""
        options = tuple(getattr(self, "active_mercenary_hire_options", ()) or ())
        if not options:
            return ()

        active_site_names = set(self._mercenary_sites_at_position(self.grid_env.agent_pos))
        if not active_site_names:
            return tuple(False for _ in options)

        hero = self._resolve_travel_hero()
        if hero is None or not self._hero_has_leadership(units=[hero]):
            return tuple(False for _ in options)

        roster_by_site_slot: Dict[Tuple[str, int], Dict[str, object]] = {}
        for site_name in active_site_names:
            for entry in self.mercenary_site_rosters.get(str(site_name), []):
                if not isinstance(entry, dict):
                    continue
                try:
                    slot = int(entry.get("slot", 0) or 0)
                except (TypeError, ValueError):
                    continue
                roster_by_site_slot[(str(site_name), int(slot))] = entry

        occupancy = self._blue_occupancy_snapshot()
        gold_available = float(self.gold or 0.0)
        first_empty_by_positions: Dict[Tuple[int, ...], Optional[int]] = {}
        values: List[bool] = []

        for option in options:
            if not isinstance(option, dict):
                values.append(False)
                continue
            site_name = str(option.get("site_name", "") or "")
            if not site_name or site_name not in active_site_names:
                values.append(False)
                continue
            try:
                slot = int(option.get("slot", 0) or 0)
            except (TypeError, ValueError):
                values.append(False)
                continue

            roster_entry = roster_by_site_slot.get((site_name, slot))
            stock = max(0, int((roster_entry or {}).get("stock", 0) or 0))
            if stock <= 0:
                values.append(False)
                continue

            unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
            unit_data = self._find_unit_data_by_name(unit_name)
            if not unit_name or unit_data is None or self._unit_data_is_big(unit_data):
                values.append(False)
                continue

            stand = str(
                (roster_entry or {}).get("stand", option.get("stand", "")) or ""
            ).strip().lower()
            if not stand:
                resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
                stand = "" if resolved_stand is None else resolved_stand
            positions = self._hire_positions_for_stand(stand)
            if not positions:
                values.append(False)
                continue

            try:
                cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                values.append(False)
                continue
            if gold_available < cost:
                values.append(False)
                continue

            if positions not in first_empty_by_positions:
                first_empty_by_positions[positions] = self._find_first_empty_blue_position(
                    positions,
                    occupancy=occupancy,
                )
            values.append(first_empty_by_positions[positions] is not None)
        return tuple(values)
    def _hire_mercenary_unit_option(
        self,
        option: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[str], Optional[int], float, Optional[str], int, int]:
        """Выполнить найм наёмника из лагеря и уменьшить stock.

        Возвращает расширенный tuple: успех, имя, позиция, потраченное золото,
        имя site, stock до и после. Эти данные затем попадают в info и метрики.
        """
        if not self._can_hire_mercenary_unit_option(option):
            return False, None, None, 0.0, None, 0, 0

        assert isinstance(option, dict)
        site_name = str(option.get("site_name", "") or "")
        slot = int(option.get("slot", 0) or 0)
        unit_name = str(option.get("unit_name", option.get("name", "")) or "").strip()
        unit_data = self._find_unit_data_by_name(unit_name)
        if unit_data is None or self._unit_data_is_big(unit_data):
            return False, None, None, 0.0, site_name or None, 0, 0

        stand = str(option.get("stand", "") or "").strip().lower()
        if not stand:
            resolved_stand = self._mercenary_stand_for_unit_data(unit_data)
            stand = "" if resolved_stand is None else resolved_stand
        positions = self._hire_positions_for_stand(stand)
        target_pos = self._find_first_empty_blue_position(positions)
        if target_pos is None:
            return False, None, None, 0.0, site_name or None, 0, 0

        roster_entry = self._mercenary_roster_entry(site_name, slot)
        stock_before = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        if stock_before <= 0:
            return False, None, None, 0.0, site_name or None, stock_before, stock_before

        recruited_unit = self._build_unit_from_data(unit_data, "blue", target_pos)
        state = self._get_blue_state()
        replaced = False
        for idx, unit in enumerate(state):
            if int(unit.get("position", -1) or -1) != int(target_pos):
                continue
            state[idx] = deepcopy(recruited_unit)
            replaced = True
            break
        if not replaced:
            state.append(deepcopy(recruited_unit))
            state.sort(key=lambda unit: int(unit.get("position", 999) or 999))

        self._clear_hero_needaunit_after_hire(state)

        cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
        self.gold = max(0.0, float(self.gold or 0.0) - cost)
        if roster_entry is not None:
            roster_entry["stock"] = max(0, stock_before - 1)
        stock_after = max(0, int(roster_entry.get("stock", 0) or 0)) if roster_entry else 0
        self._mark_equipment_dirty()
        self._sync_hero_progression_flags(state)
        self._log(
            f'Лагерь наёмников {site_name}: "{unit_name}" нанят на позицию {int(target_pos)} '
            f"(стоимость={cost:.1f} gold, gold_left={float(self.gold):.1f}, stock_left={stock_after})"
        )
        return True, unit_name, int(target_pos), cost, site_name or None, stock_before, stock_after
    def _sync_moves_per_turn_with_hero(
        self,
        units: Optional[List[Dict]] = None,
        *,
        refill: bool = False,
        grant_delta: bool = False,
    ) -> None:
        """Синхронизировать лимит очков хода с героем и экипированными ботинками.

        `refill=True` полностью восстанавливает ходы до нового cap. `grant_delta`
        добавляет только прирост cap, если бонус движения увеличился. Без этих
        флагов текущие ходы лишь обрезаются сверху при уменьшении cap.
        """
        previous_cap = int(self.moves_per_turn)
        self._sync_equipped_boot_items(units=units)
        movement_hero = self._resolve_travel_hero(units=units, alive_only=True)
        base_moves = self._hero_base_moves(units=units)
        new_cap = base_moves + self._hero_move_bonus(
            units=[movement_hero] if movement_hero is not None else [],
            base_moves=base_moves,
        )
        self.moves_per_turn = new_cap
        if refill:
            self.moves = new_cap
            return
        if grant_delta and new_cap > previous_cap:
            self.moves = min(new_cap, int(self.moves) + (new_cap - previous_cap))
        elif int(self.moves) > new_cap:
            self.moves = new_cap
    def _heal_unit_to_full_at_position(self, position: int) -> Tuple[float, Optional[str]]:
        """Лечит одного юнита BLUE на позиции в пределах золота с тарифом по Level."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or hp)

            if max_hp <= 0 or hp <= 0 or hp >= max_hp:
                return 0.0, None

            missing_hp = max_hp - hp
            gold_available = max(0.0, float(self.gold or 0.0))
            if gold_available <= 0.0:
                return 0.0, None

            gold_per_hp = self._castle_heal_gold_per_hp(unit)
            if gold_per_hp <= 0.0:
                return 0.0, None

            healed = min(missing_hp, gold_available / gold_per_hp)
            gold_spent = healed * gold_per_hp
            new_hp = hp + healed
            unit["hp"] = new_hp
            unit["health"] = new_hp
            self.gold = max(0.0, gold_available - gold_spent)

            try:
                level_for_log = int(round(float(unit.get("Level", 1) or 1)))
            except (TypeError, ValueError):
                level_for_log = 1

            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile: {name} (pos {position}) HP {hp:.1f} -> {new_hp:.1f} "
                f"(+{healed:.1f}, level={level_for_log}, "
                f"cost={gold_spent:.1f} gold, gold_left={self.gold:.1f})"
            )
            return healed, name

        return 0.0, None
    def _step_heal_in_castle(self, action: int):
        """Лечит выбранного юнита BLUE на клетках лечения с ценой за HP по Level."""
        idx = action - self.GRID_CASTLE_HEAL_ACTION_START
        target_pos = (
            self.CASTLE_HEAL_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_HEAL_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        healed_amount, unit_name = (0.0, None)
        gold_spent = 0.0
        gold_before = float(self.gold)
        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        if self._is_at_castle() and temple_built and target_pos is not None:
            healed_amount, unit_name = self._heal_unit_to_full_at_position(position=target_pos)
            gold_spent = max(0.0, gold_before - float(self.gold))
        reward = max(0.0, float(healed_amount)) * self.reward_castle_heal_per_hp

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_heal_action": True,
            "castle_pos_required": self.CASTLE_POS,
            "castle_heal_tiles": self.castle_heal_tiles,
            "castle_heal_available": self._is_at_castle() and temple_built,
            "castle_heal_target_pos": target_pos,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "healed_amount": healed_amount,
            "healed_unit_name": unit_name,
            "gold_spent": gold_spent,
            "castle_heal_reward": reward,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _ensure_map_layout_consistency(self) -> None:
        """Fail fast if generated map features overlap on forbidden tiles."""
        heal_tiles = {tuple(tile) for tile in self.castle_heal_tiles}
        enemy_tiles = {tuple(pos) for pos in self.grid_env.enemy_positions.values()}
        obstacle_tiles = {tuple(pos) for pos in self.grid_env.obstacle_positions}
        chest_tiles = {tuple(pos) for pos in getattr(self, "_static_chests", {}).keys()}

        obstacle_enemy_overlap = obstacle_tiles & enemy_tiles
        if obstacle_enemy_overlap:
            raise ValueError(
                f"Obstacle tiles overlap enemy tiles: {sorted(obstacle_enemy_overlap)}"
            )

        obstacle_heal_overlap = obstacle_tiles & heal_tiles
        if obstacle_heal_overlap:
            raise ValueError(
                f"Obstacle tiles overlap heal tiles: {sorted(obstacle_heal_overlap)}"
            )

        chest_enemy_overlap = chest_tiles & enemy_tiles
        if chest_enemy_overlap:
            raise ValueError(
                f"Chest tiles overlap enemy tiles: {sorted(chest_enemy_overlap)}"
            )

        chest_heal_overlap = chest_tiles & heal_tiles
        if chest_heal_overlap:
            raise ValueError(
                f"Chest tiles overlap heal tiles: {sorted(chest_heal_overlap)}"
            )

        chest_obstacle_overlap = chest_tiles & obstacle_tiles
        if chest_obstacle_overlap:
            raise ValueError(
                f"Chest tiles overlap obstacle tiles: {sorted(chest_obstacle_overlap)}"
            )

        reachable_targets = set(enemy_tiles)
        reachable_targets.update(heal_tiles)
        reachable_targets.update(chest_tiles)
        if not are_targets_reachable(
            self.grid_size,
            self.CASTLE_POS,
            obstacle_tiles,
            reachable_targets,
        ):
            raise ValueError("Obstacle tiles block path to at least one enemy or heal tile")
    def _castle_revive_cost_at_position(self, position: int) -> float:
        """Возвращает стоимость воскрешения мёртвого юнита на позиции или 0.0."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue
            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return 0.0
            return float(self._castle_revive_gold_cost(unit))
        return 0.0
    def _revive_unit_with_gold_at_position(
        self,
        position: int,
    ) -> Tuple[bool, Optional[str], float]:
        """Воскрешает юнита за золото на клетке лечения, если хватает средств."""
        state = self._get_blue_state()
        for unit in state:
            if int(unit.get("position", -1)) != position:
                continue

            hp = float(unit.get("hp", 0) or unit.get("health", 0))
            max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
            if max_hp <= 0 or hp > 0:
                return False, None, 0.0

            revive_cost = float(self._castle_revive_gold_cost(unit))
            gold_available = max(0.0, float(self.gold or 0.0))
            if revive_cost <= 0.0 or gold_available < revive_cost:
                return False, None, 0.0

            unit["hp"] = 1.0
            unit["health"] = 1.0
            self.gold = max(0.0, gold_available - revive_cost)
            name = unit.get("name", f"pos_{position}")
            self._log(
                f"Heal tile revive: {name} (pos {position}) revived with 1 HP "
                f"(cost={revive_cost:.1f} gold, gold_left={self.gold:.1f})"
            )
            return True, name, revive_cost

        return False, None, 0.0
    def _step_revive_in_castle(self, action: int):
        """Воскрешает выбранного юнита BLUE за золото на клетках лечения."""
        idx = action - self.GRID_CASTLE_REVIVE_ACTION_START
        target_pos = (
            self.CASTLE_REVIVE_POSITIONS[idx]
            if 0 <= idx < len(self.CASTLE_REVIVE_POSITIONS)
            else None
        )

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        revive_cost = 0.0
        target_unit_name = None
        if target_pos is not None:
            revive_cost = self._castle_revive_cost_at_position(target_pos)
            for unit in self._get_blue_state():
                if int(unit.get("position", -1)) == target_pos:
                    target_unit_name = unit.get("name", f"pos_{target_pos}")
                    break

        temple_built = self._has_temple_built()
        missing_temple = not temple_built
        revived, revived_unit_name, gold_spent = (False, None, 0.0)
        if self._is_at_castle() and temple_built and target_pos is not None:
            revived, revived_unit_name, gold_spent = self._revive_unit_with_gold_at_position(
                position=target_pos
            )
        reward = self.reward_castle_revive if revived else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "castle_revive_action": True,
            "castle_heal_tiles": self.castle_heal_tiles,
            "temple_built": temple_built,
            "missing_temple": missing_temple,
            "castle_revive_available": (
                self._is_at_castle()
                and temple_built
                and target_pos is not None
                and self._can_castle_revive_position(target_pos)
            ),
            "castle_revive_target_pos": target_pos,
            "target_unit_name": target_unit_name,
            "revived": revived,
            "revived_unit_name": revived_unit_name,
            "revive_cost": revive_cost,
            "gold_spent": gold_spent,
            "castle_revive_reward": reward,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_hire_faction_unit(self, action: int):
        """Обработать grid action фракционного найма.

        Action переводится в индекс `active_hire_options`, после чего функция
        заранее собирает диагностический контекст, выполняет hire helper и
        возвращает standard grid-step result. Reward начисляется только при
        успешном найме.
        """
        idx = action - self.GRID_HIRE_ACTION_START
        option = self._hire_option(idx)

        grid_obs = self._get_grid_obs()
        terminated = False
        truncated = False

        unit_name = None if option is None else str(option.get("name", "") or "").strip()
        hire_cost = 0.0
        hire_role = None if option is None else str(option.get("role", "") or "").strip().lower()
        hire_target_pos = None
        hire_column_positions = None
        hire_big_unit = self._is_big_hire_option(option)
        if option is not None:
            try:
                hire_cost = max(0.0, float(option.get("gold", 0.0) or 0.0))
            except (TypeError, ValueError):
                hire_cost = 0.0
            if hire_big_unit:
                hire_target_pos = self._find_first_empty_blue_column_front_position()
                if hire_target_pos is not None:
                    hire_column_positions = self._blue_column_positions_for_position(
                        int(hire_target_pos)
                    )
            else:
                positions = self._hire_positions_for_role(hire_role)
                if positions:
                    hire_target_pos = self._find_first_empty_blue_position(positions)

        hire_available_before = self._can_hire_faction_unit_option(option)
        hired, hired_unit_name, hired_position, gold_spent = self._hire_faction_unit_option(option)

        hero = self._resolve_travel_hero()
        hero_needaunit = self._resolve_hero_needaunit(hero) if hero is not None else 0
        reward = self._hire_reward_value() if hired else 0.0

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "hire_action": True,
            "hire_action_idx": int(idx),
            "hire_unit_name": unit_name,
            "hire_role": hire_role,
            "hire_big_unit": bool(hire_big_unit),
            "hire_cost": float(hire_cost),
            "hire_target_pos": hire_target_pos,
            "hire_column_positions": hire_column_positions,
            "hire_available": bool(hire_available_before),
            "hired": bool(hired),
            "hired_units_count": int(1 if hired else 0),
            "hired_unit_name": hired_unit_name,
            "hired_position": hired_position,
            "gold_spent": float(gold_spent),
            "hire_reward": float(reward),
            "hero_needaunit": int(hero_needaunit),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _normalize_capital_id(self, value: int) -> int:
        """Нормализует Realcapital: ожидает целое 1–5, иначе возвращает 1 (значение по умолчанию)."""
        try:
            capital_id = int(value)
        except (TypeError, ValueError):
            return 1
        return capital_id if capital_id in (1, 2, 3, 4, 5) else 1
    def _normalize_typeoflord(self, value: int) -> int:
        """Нормализовать тип лорда к поддерживаемому диапазону 1-3.

        1 соответствует воину, 2 магу, 3 вору/разведчику. Некорректные значения
        падают в 1, чтобы downstream unlock'и и building modifiers не получали
        неизвестный класс.
        """
        try:
            lord_type = int(value)
        except (TypeError, ValueError):
            return 1
        return lord_type if lord_type in (1, 2, 3) else 1
    @staticmethod
    def _clip01(value: float) -> float:
        """Обрезать числовое значение в диапазон [0, 1] для observation features."""
        return float(np.clip(float(value), 0.0, 1.0))
    def _get_building_keys(self, buildings: Dict) -> List[str]:
        """Список ключей построек (без служебного ключа 'alredybuilt')."""
        active_metadata = getattr(self, "_active_building_table_metadata", {})
        if isinstance(active_metadata, dict) and active_metadata.get("keys"):
            keys = [str(key) for key in active_metadata.get("keys", ())]
            metadata_by_key = active_metadata.get("metadata_by_key", {})
            key_by_name = active_metadata.get("key_by_name", {})
            self._building_metadata_by_key = (
                metadata_by_key if isinstance(metadata_by_key, dict) else {}
            )
            self._building_key_by_name = key_by_name if isinstance(key_by_name, dict) else {}
            self._building_requirement_keys_by_key = {
                str(key): tuple(metadata.get("requirement_keys", ()) or ())
                for key, metadata in self._building_metadata_by_key.items()
                if isinstance(metadata, dict)
            }
            self._building_block_keys_by_key = {
                str(key): tuple(metadata.get("block_keys", ()) or ())
                for key, metadata in self._building_metadata_by_key.items()
                if isinstance(metadata, dict)
            }
            self._grid_max_build_price_cache_signature = None
            self._grid_max_build_price_cache = 1.0
            return keys

        keys: List[str] = []
        metadata_by_key: Dict[str, Dict[str, object]] = {}
        key_by_name: Dict[str, str] = {}
        for key, entry in buildings.items():
            if isinstance(entry, dict) and entry.get("id"):
                key_s = str(key)
                keys.append(key_s)
                name = str(entry.get("name", "") or "").strip()
                requirements = self._building_name_tuple(entry.get("requires", ()))
                blocks = self._building_name_tuple(entry.get("blocks", ()))
                metadata_by_key[key_s] = {
                    "name": name,
                    "requires": requirements,
                    "blocks": blocks,
                }
                if name:
                    key_by_name[name] = key_s
        self._building_metadata_by_key = metadata_by_key
        self._building_key_by_name = key_by_name
        self._building_requirement_keys_by_key = {}
        self._building_block_keys_by_key = {}
        self._grid_max_build_price_cache_signature = None
        self._grid_max_build_price_cache = 1.0
        return keys
    def _reset_buildings_state(self) -> None:
        """Создать локальные таблицы зданий для текущего CampaignEnv."""
        self._buildings_by_capital_key = {}
        self._building_max_gold_by_capital_key = {}
        cost_multiplier = max(0.0, float(self.BUILDING_GOLD_COST_MULTIPLIER))
        for key, template in BUILDINGS_D2_TEMPLATE.items():
            if not isinstance(template, dict):
                continue
            table_key = str(key)
            metadata = BUILDING_METADATA_D2.get(table_key, {})
            building_keys = tuple(
                metadata.get("keys", ())
                or (
                    str(build_key)
                    for build_key, entry in template.items()
                    if isinstance(entry, dict) and entry.get("id")
                )
            )
            buildings: Dict[str, object] = {
                "alredybuilt": template.get("alredybuilt", 0),
            }
            max_gold = 1.0
            for build_key in building_keys:
                entry = template.get(build_key)
                if not isinstance(entry, dict):
                    continue
                building = entry.copy()
                if cost_multiplier != 1.0 and "gold" in building:
                    try:
                        base_gold = float(building.get("gold", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        base_gold = 0.0
                    building["gold"] = base_gold * cost_multiplier
                try:
                    max_gold = max(max_gold, float(building.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass
                buildings[str(build_key)] = building
            self._buildings_by_capital_key[table_key] = buildings
            self._building_max_gold_by_capital_key[table_key] = float(max_gold)
    @classmethod
    def _scale_building_gold_costs_in_place(cls, buildings: Dict) -> None:
        """Применить глобальный множитель стоимости зданий к таблице in-place.

        Таблицы зданий копируются из шаблона при reset, поэтому изменение
        локальной копии безопасно. Если множитель равен 1, функция ничего не
        делает, чтобы не трогать исходные значения.
        """
        cost_multiplier = max(0.0, float(cls.BUILDING_GOLD_COST_MULTIPLIER))
        if cost_multiplier == 1.0:
            return

        for entry in buildings.values():
            if not isinstance(entry, dict) or "gold" not in entry:
                continue
            try:
                base_gold = float(entry.get("gold", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            entry["gold"] = base_gold * cost_multiplier
    def _building_cost_multiplier(self) -> float:
        """Вернуть классовый множитель стоимости зданий.

        Для `typeoflord == 3` используется отдельный коэффициент, остальные
        типы лорда строят здания по базовой цене.
        """
        return (
            float(self.TYPEOFLORD_THREE_BUILDING_COST_MULTIPLIER)
            if int(self.typeoflord) == 3
            else 1.0
        )
    def _get_building_gold_cost(self, building: Optional[Dict[str, object]]) -> float:
        """Посчитать итоговую цену здания с учётом классового множителя."""
        if not isinstance(building, dict):
            return 0.0
        try:
            base_cost = max(0.0, float(building.get("gold", 0.0) or 0.0))
        except (TypeError, ValueError):
            base_cost = 0.0
        return base_cost * float(self._building_cost_multiplier())
    def _current_max_building_gold_cost(self) -> float:
        """Вернуть максимальную цену среди активных зданий текущей фракции.

        Значение используется для нормализации observation, поэтому минимум
        равен 1.0 и деление на ноль невозможно даже при пустой таблице зданий.
        """
        signature = (id(self.active_buildings), int(self.typeoflord))
        if getattr(self, "_grid_max_build_price_cache_signature", None) == signature:
            return float(getattr(self, "_grid_max_build_price_cache", 1.0) or 1.0)

        max_price = 1.0
        try:
            active_max_gold = float(getattr(self, "_active_building_max_gold", 0.0) or 0.0)
        except (TypeError, ValueError):
            active_max_gold = 0.0
        if active_max_gold > 0.0:
            max_price = max(
                1.0,
                active_max_gold * float(self._building_cost_multiplier()),
            )
        else:
            for build_key in self.building_keys:
                max_price = max(
                    max_price,
                    self._get_building_gold_cost(self.active_buildings.get(build_key)),
                )
        self._grid_max_build_price_cache_signature = signature
        self._grid_max_build_price_cache = float(max_price)
        self.grid_max_build_price = float(max_price)
        return float(max_price)
    def get_built_building_names(self) -> List[str]:
        """Возвращает названия построенных зданий активной фракции в порядке ключей."""
        built_names: List[str] = []
        for key in self.building_keys:
            building = self.active_buildings.get(key)
            if not isinstance(building, dict):
                continue
            built_flag = building.get("Build", building.get("built", 0))
            try:
                is_built = int(built_flag or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            if not is_built:
                continue
            name = str(building.get("name", "") or "").strip()
            if name:
                built_names.append(name)
        return built_names
    def _built_building_names_set(self) -> set[str]:
        """Вернуть set названий уже построенных зданий активной фракции."""
        built_names: set[str] = set()
        metadata_by_key = getattr(self, "_building_metadata_by_key", {})
        for key in self.building_keys:
            building = self.active_buildings.get(key)
            if not isinstance(building, dict):
                continue
            built_flag = building.get("Build", building.get("built", 0))
            try:
                is_built = int(built_flag or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            if not is_built:
                continue
            metadata = metadata_by_key.get(str(key), {}) if isinstance(metadata_by_key, dict) else {}
            name = str(metadata.get("name", building.get("name", "")) or "").strip()
            if name:
                built_names.add(name)
        return built_names
    def _built_building_keys_set(self) -> set[str]:
        built_keys: set[str] = set()
        for key in self.building_keys:
            building = self.active_buildings.get(key)
            if not isinstance(building, dict):
                continue
            built_flag = building.get("Build", building.get("built", 0))
            try:
                is_built = int(built_flag or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            if is_built:
                built_keys.add(str(key))
        return built_keys
    @staticmethod
    def _building_name_tuple(raw_names: object) -> Tuple[str, ...]:
        """Нормализовать поле requires/blocks здания в tuple имён."""
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
    def _building_metadata_for_entry(
        self,
        building: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        if not isinstance(building, dict):
            return {}
        metadata_by_key = getattr(self, "_building_metadata_by_key", {})
        building_key = str(building.get("id", "") or "")
        if isinstance(metadata_by_key, dict):
            metadata = metadata_by_key.get(building_key, {})
            if isinstance(metadata, dict):
                return metadata
        return {}
    def _building_requirement_names(
        self,
        building: Optional[Dict[str, object]],
    ) -> Tuple[str, ...]:
        """Нормализовать список prerequisite-зданий из поля requires."""
        if not isinstance(building, dict):
            return ()
        metadata = self._building_metadata_for_entry(building)
        if metadata:
            return tuple(metadata.get("requires", ()) or ())
        return self._building_name_tuple(building.get("requires", ()))
    def _building_block_names(
        self,
        building: Optional[Dict[str, object]],
    ) -> Tuple[str, ...]:
        """Вернуть список зданий, блокируемых этой постройкой."""
        if not isinstance(building, dict):
            return ()
        metadata = self._building_metadata_for_entry(building)
        if metadata:
            return tuple(metadata.get("blocks", ()) or ())
        return self._building_name_tuple(building.get("blocks", ()))
    def _building_requirement_keys(
        self,
        building: Optional[Dict[str, object]],
    ) -> Tuple[str, ...]:
        if not isinstance(building, dict):
            return ()
        metadata = self._building_metadata_for_entry(building)
        if metadata:
            return tuple(str(key) for key in (metadata.get("requirement_keys", ()) or ()))
        key_by_name = getattr(self, "_building_key_by_name", {})
        if not isinstance(key_by_name, dict):
            return ()
        return tuple(
            str(key_by_name[name])
            for name in self._building_name_tuple(building.get("requires", ()))
            if name in key_by_name
        )
    def _building_block_keys(
        self,
        building: Optional[Dict[str, object]],
    ) -> Tuple[str, ...]:
        if not isinstance(building, dict):
            return ()
        metadata = self._building_metadata_for_entry(building)
        if metadata:
            return tuple(str(key) for key in (metadata.get("block_keys", ()) or ()))
        key_by_name = getattr(self, "_building_key_by_name", {})
        if not isinstance(key_by_name, dict):
            return ()
        return tuple(
            str(key_by_name[name])
            for name in self._building_name_tuple(building.get("blocks", ()))
            if name in key_by_name
        )
    def _missing_building_requirements(
        self,
        building: Optional[Dict[str, object]],
        *,
        built_names: Optional[set[str]] = None,
    ) -> Tuple[str, ...]:
        """Вернуть prerequisite-здания, которые ещё не построены."""
        requirement_names = self._building_requirement_names(building)
        if not requirement_names:
            return ()
        if built_names is None:
            built_names = self._built_building_names_set()
        return tuple(name for name in requirement_names if name not in built_names)
    def _missing_building_requirement_keys(
        self,
        building: Optional[Dict[str, object]],
        *,
        built_keys: Optional[set[str]] = None,
    ) -> Tuple[str, ...]:
        requirement_keys = self._building_requirement_keys(building)
        requirement_names = self._building_requirement_names(building)
        if not requirement_names:
            return ()
        if len(requirement_keys) != len(requirement_names):
            missing_names = self._missing_building_requirements(building)
            return tuple(f"name:{name}" for name in missing_names)
        if built_keys is None:
            built_keys = self._built_building_keys_set()
        return tuple(key for key in requirement_keys if key not in built_keys)
    def _building_action_mask_values(self) -> Tuple[bool, ...]:
        """Batch-доступность всех building actions."""
        try:
            has_build_lock = int(self.active_buildings.get("alredybuilt", 0) or 0) == 1
        except (TypeError, ValueError):
            has_build_lock = False
        built_keys = self._built_building_keys_set()
        gold_available = float(self.gold or 0.0)
        values: List[bool] = []
        for build_key in self.building_keys:
            building = self.active_buildings.get(build_key)
            if not isinstance(building, dict):
                values.append(False)
                continue
            try:
                is_built = int(building.get("Build", building.get("built", 0)) or 0) == 1
            except (TypeError, ValueError):
                is_built = False
            try:
                is_blocked = int(building.get("blocked", 0) or 0) == 1
            except (TypeError, ValueError):
                is_blocked = False
            missing_requirements = self._missing_building_requirement_keys(
                building,
                built_keys=built_keys,
            )
            values.append(
                not has_build_lock
                and not is_built
                and not is_blocked
                and not missing_requirements
                and gold_available >= self._get_building_gold_cost(building)
            )
        return tuple(values)
    def _step_building(self, action: int):
        """Выбирает здание активной фракции и отмечает его как построенное."""
        idx = action - self.GRID_BUILD_ACTION_START
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        build_key = None
        build_name = None
        built = False
        already_built = False
        blocked_names: List[str] = []
        build_cost = None
        insufficient_gold = False
        build_blocked = False
        build_locked = False
        build_requirement_names: Tuple[str, ...] = ()
        missing_requirements: Tuple[str, ...] = ()
        requirements_missing = False

        if 0 <= idx < len(self.building_keys):
            build_key = self.building_keys[idx]
            building = self.active_buildings.get(build_key)
            if isinstance(building, dict):
                build_name = building.get("name")
                build_cost = self._get_building_gold_cost(building)
                alredybuilt_flag = self.active_buildings.get("alredybuilt", 0)
                try:
                    build_locked = int(alredybuilt_flag or 0) == 1
                except (TypeError, ValueError):
                    build_locked = False
                built_flag = building.get("Build", building.get("built", 0))
                already_built = int(built_flag or 0) == 1
                blocked_flag = building.get("blocked", 0)
                build_blocked = int(blocked_flag or 0) == 1
                build_requirement_names = self._building_requirement_names(building)
                missing_requirement_keys = self._missing_building_requirement_keys(
                    building,
                    built_keys=self._built_building_keys_set(),
                )
                missing_requirements = self._missing_building_requirements(building)
                requirements_missing = bool(missing_requirement_keys)
                if build_locked:
                    pass
                elif build_blocked:
                    pass
                elif requirements_missing:
                    pass
                elif self.gold < (build_cost or 0):
                    insufficient_gold = True
                elif not already_built:
                    self.gold -= build_cost or 0
                    building["built"] = 1
                    built = True
                    self.active_buildings["alredybuilt"] = 1

                    block_names = self._building_block_names(building)
                    block_keys = self._building_block_keys(building)

                    if block_keys and len(block_keys) == len(block_names):
                        for blocked_key in block_keys:
                            blocked_entry = self.active_buildings.get(blocked_key)
                            if not isinstance(blocked_entry, dict):
                                continue
                            blocked_entry["blocked"] = 1
                            blocked_name = str(blocked_entry.get("name", "") or "").strip()
                            if blocked_name:
                                blocked_names.append(blocked_name)
                    elif block_names:
                        for entry in self.active_buildings.values():
                            if not isinstance(entry, dict):
                                continue
                            entry_name = str(entry.get("name", "") or "").strip()
                            if entry_name and entry_name in block_names:
                                entry["blocked"] = 1
                                blocked_names.append(entry_name)

                    if build_name:
                        self._log("ЗДАНИЕ ПОСТРОЕНО")
                        self._log(f'Построено здание "{build_name}"')
                        if blocked_names:
                            self._log(f'Заблокированы здания: {", ".join(blocked_names)}')

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "build_action": True,
            "build_key": build_key,
            "build_name": build_name,
            "built": built,
            "already_built": already_built,
            "blocked_names": blocked_names,
            "build_cost": build_cost,
            "insufficient_gold": insufficient_gold,
            "build_blocked": build_blocked,
            "build_locked": build_locked,
            "build_requires": list(build_requirement_names),
            "build_missing_requirements": list(missing_requirements),
            "build_requirements_met": not requirements_missing,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _step_upgrade_settlement(self, action: int):
        """Обработать action улучшения захваченного поселения.

        Улучшение доступно только для поселения из layout, которое уже захвачено,
        не достигло max level и имеет оплачиваемую стоимость перехода. Функция
        списывает золото, повышает уровень и возвращает info с причиной успеха
        или блокировки.
        """
        idx = action - self.grid_settlement_upgrade_action_start
        grid_obs = self._get_grid_obs()
        reward = 0.0
        terminated = False
        truncated = False

        settlement_name = None
        source_tile = None
        captured = False
        current_level = None
        next_level = None
        upgrade_cost = 0.0
        upgraded = False
        insufficient_gold = False
        at_max_level = False

        if 0 <= idx < len(self.settlement_upgrade_names):
            settlement_name = str(self.settlement_upgrade_names[idx])
            source_tile = self.legions_settlement_source_tile_by_name.get(settlement_name)
            captured = (
                settlement_name in self.legions_active_settlement_territory_capture_turn_by_name
            )
            current_level = int(self.legions_settlement_level_by_name.get(settlement_name, 1) or 1)
            at_max_level = current_level >= int(self.MAX_SETTLEMENT_LEVEL)
            upgrade_cost = self._settlement_upgrade_cost_for_level(current_level)
            insufficient_gold = (
                captured
                and not at_max_level
                and upgrade_cost > 0.0
                and float(self.gold or 0.0) < float(upgrade_cost)
            )
            next_level = min(int(self.MAX_SETTLEMENT_LEVEL), current_level + 1)
            if captured and not at_max_level and upgrade_cost > 0.0 and not insufficient_gold:
                self.gold -= float(upgrade_cost)
                self.legions_settlement_level_by_name[settlement_name] = int(next_level)
                upgraded = True
                self._log(
                    f'Город "{settlement_name}" улучшен: уровень {current_level} -> {next_level} '
                    f"за {upgrade_cost:g} золота."
                )

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": False,
            "settlement_upgrade_action": True,
            "settlement_upgrade_name": settlement_name,
            "settlement_upgrade_source_tile": tuple(source_tile) if source_tile is not None else None,
            "settlement_upgrade_captured": bool(captured),
            "settlement_upgrade_old_level": current_level,
            "settlement_upgrade_new_level": next_level if upgraded else current_level,
            "settlement_upgrade_cost": float(upgrade_cost),
            "settlement_upgrade_applied": bool(upgraded),
            "settlement_upgrade_insufficient_gold": bool(insufficient_gold),
            "settlement_upgrade_at_max_level": bool(at_max_level),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
        }

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _building_key_for_capital(self, capital: Optional[int]) -> str:
        """Преобразовать id столицы в ключ локальной таблицы зданий."""
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            return "legions"
        if capital_value == 3:
            return "mountain_clans"
        if capital_value == 4:
            return "undead_hordes"
        if capital_value == 5:
            return "elves"
        return "empire"
    def _get_buildings_for_capital(self, capital: Optional[int]) -> Dict:
        """Вернуть локальную таблицу построек для выбранной столицы."""
        building_key = self._building_key_for_capital(capital)
        tables = getattr(self, "_buildings_by_capital_key", None)
        if not isinstance(tables, dict) or building_key not in tables:
            self._reset_buildings_state()
            tables = self._buildings_by_capital_key
        self._active_building_capital_key = building_key
        self._active_building_table_metadata = BUILDING_METADATA_D2.get(building_key, {})
        max_gold_by_capital = getattr(self, "_building_max_gold_by_capital_key", {})
        self._active_building_max_gold = float(max_gold_by_capital.get(building_key, 1.0) or 1.0)
        return tables[building_key]
    def _get_hire_options_for_capital(self, capital: Optional[int]) -> List[Dict[str, object]]:
        """Вернуть список фракционных вариантов найма для выбранной столицы.

        Источник выбирается по `Realcapital`: Empire, Legions, Mountain Clans,
        Undead Hordes или Elves. В конец списка добавляется synthetic big-unit
        option, если для этой фракции он определён.
        """
        try:
            capital_value = int(capital)
        except (TypeError, ValueError):
            capital_value = None

        if capital_value == 2:
            hire_data = HIRE_D2["legions"]
        elif capital_value == 3:
            hire_data = HIRE_D2["mountain_clans"]
        elif capital_value == 4:
            hire_data = HIRE_D2["undead_hordes"]
        elif capital_value == 5:
            hire_data = HIRE_D2["elves"]
        else:
            hire_data = HIRE_D2["empire"]

        if not isinstance(hire_data, dict):
            return []
        options = [dict(entry) for entry in hire_data.values() if isinstance(entry, dict)]
        big_option = self._faction_big_hire_option_for_capital(capital_value)
        if big_option is not None:
            options.append(big_option)
        return options
    def _get_mercenary_hire_options(self) -> List[Dict[str, object]]:
        """Собрать плоский список всех вариантов найма из лагерей наёмников.

        Каждый option хранит site, slot, имя юнита, цену, stand и тип. Stock не
        копируется сюда намеренно: актуальный остаток читается из site roster
        при проверке и выполнении найма.
        """
        options: List[Dict[str, object]] = []
        for site_name, site_data in self.BASE_MERCENARY_SITE_DATA.items():
            site_id = str(site_data.get("site_id", "") or "")
            for idx, entry in enumerate(tuple(site_data.get("roster", ()))):
                if not isinstance(entry, dict):
                    continue
                unit_name = str(entry.get("unit_name", entry.get("name", "")) or "").strip()
                if not unit_name:
                    continue
                slot = int(entry.get("slot", idx) or idx)
                try:
                    gold_cost = max(0.0, float(entry.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    gold_cost = 0.0
                unit_data = self._find_unit_data_by_name(unit_name)
                stand = self._mercenary_stand_for_unit_data(unit_data)
                options.append(
                    {
                        "site_name": str(site_name),
                        "site_id": site_id,
                        "slot": int(slot),
                        "unit_id": str(entry.get("unit_id", "") or ""),
                        "unit_name": unit_name,
                        "name": unit_name,
                        "gold": float(gold_cost),
                        "stand": "" if stand is None else stand,
                        "unit_type": (
                            str(unit_data.get("тип", unit_data.get("unit_type", "")) or "").strip()
                            if isinstance(unit_data, dict)
                            else ""
                        ),
                    }
                )
        return options
    def _refresh_dynamic_action_layout(self) -> None:
        """Пересобрать все динамические диапазоны grid actions.

        Размер action space зависит от сценарных зелий, hire options, зданий,
        заклинаний, книг, боевых предметов, staff/scroll actions и swap layout.
        Функция должна вызываться после изменения сценарных списков или состава,
        чтобы action mask, step dispatch и observation использовали одинаковые
        стартовые индексы.
        """
        self.active_hire_options = self._get_hire_options_for_capital(self.Realcapital)
        self.active_mercenary_hire_options = self._get_mercenary_hire_options()
        self.GRID_POTION_USE_ACTION_START = 9
        self._refresh_potion_item_names()
        self._refresh_battle_equippable_item_names()
        self._refresh_book_item_names()
        self._refresh_scroll_item_names()
        self._refresh_staff_spell_action_entries()
        self.GRID_BOTTLE_POSITIONS = list(self.GRID_POTION_USE_POSITIONS)
        self.REVIVE_BOTTLE_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.INVULNERABILITY_POTION_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.STRENGTH_POTION_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.ENERGY_ELIXIR_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.HASTE_ELIXIR_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.FIRE_WARD_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.EARTH_WARD_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.WATER_WARD_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.AIR_WARD_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.TITAN_ELIXIR_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.SUPREME_ELIXIR_POSITIONS = self.GRID_BOTTLE_POSITIONS
        self.GRID_BOTTLE_ACTION_START = self._potion_action_start_for_item(
            self.BONUS_SMALL_HEAL_ITEM_NAME
        )
        self.GRID_REVIVE_ACTION_START = self._potion_action_start_for_item(
            self.BONUS_REVIVE_ITEM_NAME
        )
        self.GRID_INVULNERABILITY_ACTION_START = self._potion_action_start_for_item(
            self.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME
        )
        self.GRID_STRENGTH_ACTION_START = self._potion_action_start_for_item(
            self.STRENGTH_POTION_ITEM_NAME
        )
        self.GRID_ENERGY_ACTION_START = self._potion_action_start_for_item(
            self.ENERGY_ELIXIR_ITEM_NAME
        )
        self.GRID_HASTE_ACTION_START = self._potion_action_start_for_item(
            self.HASTE_ELIXIR_ITEM_NAME
        )
        self.GRID_FIRE_WARD_ACTION_START = self._potion_action_start_for_item(
            self.FIRE_WARD_ITEM_NAME
        )
        self.GRID_EARTH_WARD_ACTION_START = self._potion_action_start_for_item(
            self.EARTH_WARD_ITEM_NAME
        )
        self.GRID_WATER_WARD_ACTION_START = self._potion_action_start_for_item(
            self.WATER_WARD_ITEM_NAME
        )
        self.GRID_AIR_WARD_ACTION_START = self._potion_action_start_for_item(
            self.AIR_WARD_ITEM_NAME
        )
        self.GRID_TITAN_ELIXIR_ACTION_START = self._potion_action_start_for_item(
            self.TITAN_ELIXIR_ITEM_NAME
        )
        self.GRID_SUPREME_ELIXIR_ACTION_START = self._potion_action_start_for_item(
            self.SUPREME_ELIXIR_ITEM_NAME
        )
        self.GRID_CASTLE_HEAL_ACTION_START = (
            self.GRID_POTION_USE_ACTION_START + self.GRID_POTION_USE_ACTION_COUNT
        )
        self.GRID_CASTLE_REVIVE_ACTION_START = (
            self.GRID_CASTLE_HEAL_ACTION_START + len(self.CASTLE_HEAL_POSITIONS)
        )
        self.GRID_HIRE_ACTION_START = (
            self.GRID_CASTLE_REVIVE_ACTION_START + len(self.CASTLE_REVIVE_POSITIONS)
        )
        self.GRID_MERCENARY_HIRE_ACTION_START = (
            self.GRID_HIRE_ACTION_START + len(self.active_hire_options)
        )
        self.GRID_TRAINER_ACTION_START = (
            self.GRID_MERCENARY_HIRE_ACTION_START + len(self.active_mercenary_hire_options)
        )
        self.GRID_MERCHANT_POTION_BUY_ACTION_START = (
            self.GRID_TRAINER_ACTION_START + len(self.TRAINER_POSITIONS)
        )
        self.GRID_MERCHANT_BUY_ACTION_START = self.GRID_MERCHANT_POTION_BUY_ACTION_START
        self.GRID_SPELL_SHOP_BUY_ACTION_START = (
            self.GRID_MERCHANT_POTION_BUY_ACTION_START
            + self.GRID_MERCHANT_POTION_BUY_ACTION_COUNT
        )
        self.GRID_BUILD_ACTION_START = self.GRID_SPELL_SHOP_BUY_ACTION_START + len(
            self.SPELL_SHOP_BUY_SPELLS
        )
        self.grid_spell_action_start = self.GRID_BUILD_ACTION_START + len(self.building_keys)
        self.grid_settlement_upgrade_action_start = self.grid_spell_action_start + len(
            self.spell_keys
        )
        self.GRID_EQUIP_BATTLE_POTION_ACTION_START = (
            self.grid_settlement_upgrade_action_start + len(self.settlement_upgrade_names)
        )
        self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START = self.GRID_EQUIP_BATTLE_POTION_ACTION_START
        self.GRID_EQUIP_BATTLE_ITEM2_ACTION_START = (
            self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.GRID_EQUIP_BOOK_ACTION_START = (
            self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + len(self.BATTLE_EQUIPPABLE_ITEM_NAMES)
        )
        self.grid_legion_damage_spell_action_start = (
            self.GRID_EQUIP_BOOK_ACTION_START + len(self.scenario_book_item_names)
        )
        self.grid_map_support_spell_action_start = (
            self.grid_legion_damage_spell_action_start + self._map_offensive_spell_action_max_count()
        )
        self.grid_spell_shop_cast_action_start = (
            self.grid_map_support_spell_action_start + self._map_support_spell_action_max_count()
        )
        self.grid_staff_spell_action_start = (
            self.grid_spell_shop_cast_action_start + int(self.MAX_SPELL_SHOP_CAST_ACTIONS)
        )
        self.GRID_UNLOCK_SCROLL_MAGIC_ACTION = (
            self.grid_staff_spell_action_start + len(self.staff_spell_action_entries())
        )
        self.grid_scroll_cast_action_start = self.GRID_UNLOCK_SCROLL_MAGIC_ACTION + 1
        self.GRID_SWAP_UNIT_ACTION_START = (
            self.grid_scroll_cast_action_start + int(self.GRID_SCROLL_CAST_ACTION_COUNT)
        )
        starting_blue_team = (
            self.blue_team_state
            if getattr(self, "blue_team_state", None) is not None
            else self._create_starting_blue_team()
        )
        swap_positions = self._occupied_initial_blue_positions(starting_blue_team)
        self.GRID_SWAP_UNIT_PAIRS = self._build_blue_unit_swap_pairs(
            swap_positions,
            self._extra_blue_unit_swap_positions(swap_positions),
        )
        self.GRID_SWAP_UNIT_ACTION_COUNT = len(self.GRID_SWAP_UNIT_PAIRS)
        self._refresh_book_observation_layout()
    def _spend_moves(self, spent_moves: int) -> None:
        """Списывает очки перемещения в grid-режиме."""
        if spent_moves <= 0:
            return
        self.moves = max(0, int(self.moves) - int(spent_moves))
    def _reset_stagnation_tracking(self, start_pos: Tuple[int, int]) -> None:
        """Сбрасывает трекинг повторов и микро-циклов в grid-режиме."""
        pos = (int(start_pos[0]), int(start_pos[1]))
        self.grid_visit_counts = {pos: 1}
        self.recent_positions = [pos]
    def _compute_stagnation_penalty(
        self,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
    ) -> float:
        """
        Штрафует стагнацию на карте:
        - повторные посещения клетки (с кепом на шаг),
        - движение "в стену" без смены позиции,
        - петля A->B->A.
        Возвращает отрицательное значение (или 0.0).
        """
        target_pos = (int(new_pos[0]), int(new_pos[1]))
        visit_count = int(self.grid_visit_counts.get(target_pos, 0)) + 1
        self.grid_visit_counts[target_pos] = visit_count

        self.recent_positions.append(target_pos)
        if len(self.recent_positions) > 3:
            self.recent_positions = self.recent_positions[-3:]

        penalty = 0.0
        if visit_count > 1 and self.reward_repeat_position_penalty > 0.0:
            repeat_penalty = self.reward_repeat_position_penalty * float(visit_count - 1)
            penalty += min(repeat_penalty, self.reward_repeat_position_penalty_cap)

        if old_pos == new_pos and self.reward_no_movement_penalty > 0.0:
            penalty += self.reward_no_movement_penalty

        if len(self.recent_positions) >= 3 and self.reward_backtrack_penalty > 0.0:
            pos_a, pos_b, pos_c = self.recent_positions[-3:]
            if pos_a == pos_c and pos_a != pos_b:
                penalty += self.reward_backtrack_penalty

        return -float(penalty)
