# campaign_env.py
"""
Двухуровневая среда: Grid World + Battle.
ВЕРСИЯ БЕЗ ВОССТАНОВЛЕНИЯ HP.

Агент перемещается по карте 48×48 (по умолчанию), встречает врагов и сражается с ними.
Список врагов и их позиции соответствуют совпавшим отрядам из сценарной карты.

После каждой победы HP юнитов BLUE НЕ восстанавливается,
погибшие юниты НЕ воскрешаются. Только сбрасываются временные эффекты.

Схема наград:
  - За участие в битве (нашёл врага): +0.1
  - За победу в каждой битве: +0.3
  - За полную зачистку карты: финал кампании
  - За поражение: -1.0
  - За таймаут (лимит шагов): -6.0
"""

import gymnasium as gym

from campaign_env_data import *
from campaign_env_battle import CampaignBattleMixin
from campaign_env_economy import CampaignEconomyMixin
from campaign_env_inventory import CampaignInventoryMixin
from campaign_env_magic import CampaignMagicMixin
from campaign_env_map import CampaignMapSitesMixin
from campaign_env_masks import CampaignMaskMixin
from campaign_env_observation import CampaignObservationMixin
from campaign_env_scripted_bot import CampaignScriptedBotMixin
from campaign_env_territory import CampaignTerritoryMixin


class CampaignEnv(
    CampaignConstantsMixin,
    CampaignObservationMixin,
    CampaignMaskMixin,
    CampaignBattleMixin,
    CampaignMagicMixin,
    CampaignInventoryMixin,
    CampaignEconomyMixin,
    CampaignMapSitesMixin,
    CampaignScriptedBotMixin,
    CampaignTerritoryMixin,
    gym.Env,
):
    MAX_GRID_UNIT_SWAPS_PER_TURN = 3

    """
    Двухуровневая среда: Grid World + Battle.

    Режимы:
        - MODE_GRID: агент перемещается по карте
        - MODE_BATTLE: агент сражается (используется BattleEnv)

    Награды:
        - Награда за участие в битве (нашёл врага): +0.1
        - Награда за победу в каждой битве: +0.3
        - Награда за успешный найм юнита после лидерства: 3 * reward_defeat_enemy
        - Награда за первый захваченный целевой город: reward_all_enemies
        - Награда за второй захваченный целевой город: reward_all_enemies * 5
        - Победа в кампании после захвата Порта Полонис и Соругириллы
        - В dragon-режиме: одноразовая награда за достижение зеленого дракона,
          увеличенная награда за победу над ним и нарастающий штраф за затяжку без победы
        - Масштабированная награда из BattleEnv: battle_reward_scale * battle_reward
- Награда за лечение на специальных клетках: reward_castle_heal_per_hp * restored_hp
- Награда за успешное воскрешение на специальных клетках: reward_castle_revive
- Небольшая награда за бой основного отряда героя в тот же ход после призыва юнита
        - Минимальная награда за продажу бесполезного предмета у торговца
        - Дополнительный штраф за каждый завершённый ход, пока герою нужен юнит
        - Штраф за поражение: -1.0
        - Штраф за таймаут (лимит шагов на карте): -6.0

    Observation Space:
        Объединённое наблюдение:
        - [0]: режим (0 = grid, 1 = battle)
        - [1:1+GRID_OBS_SIZE]: grid observation
          (базовые признаки + все enemy stacks + BLUE roster + ресурсы/инвентарь
           + постройки + сундуки + клетки лечения + руины + прогресс по целевым городам)
        - [...]: battle observation (когда в бою) или нули
        - [...]: ход и золото

    Action Space:
        Discrete(...):
        - В режиме grid:
          - 0-7: движение в 8 направлениях
          - 8: отдых (REST) — восстановление 5% HP раненым юнитам
            (+ бонус лечения на клетках столицы и поселений)
          - 9-14: применить подходящую банку лечения к позициям BLUE 7-12
            (автовыбор +50 HP или +100 HP в зависимости от недостающего здоровья и запасов)
          - 15-20: применить бутыль воскрешения к позициям BLUE 7-12 (оживляет с 1 HP)
          - 21-26: применить зелье неуязвимости к позициям BLUE 7-12 (+50 брони на 1 ход)
          - 27-32: применить зелье силы к позициям BLUE 7-12 (+30% урона на 1 ход)
          - 33-38: лечение на специальных клетках карты по позициям BLUE 7-12
            (стоимость за 1 HP зависит от Level)
          - 39-44: воскрешение на специальных клетках карты по позициям BLUE 7-12
            (стоимость зависит от Level)
          - 45-54: покупка товаров у торговца
          - 55-60: применить Эликсир энергии к позициям 7-12
          - 61-66: применить Эликсир быстроты к позициям 7-12
          - 67-72: применить защиту от магии Огня к позициям 7-12
          - 73-78: применить защиту от магии Земли к позициям 7-12
          - 79-84: применить защиту от магии Воды к позициям 7-12
          - 85-90: применить защиту от магии Воздуха к позициям 7-12
          - 91-96: применить Эликсир Силы титана к позициям 7-12
          - 97-102: применить Эликсир Всевышнего к позициям 7-12
          - 103-...: постройка зданий активной фракции
        - В режиме battle: используются действия 0-14 (атака/защита/ожидание)

        Action masking применяется для блокировки недопустимых действий.
    """

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        reward_engage_battle: float = 0.2,  # Бонус за вход в бой с живым врагом.
        reward_defeat_enemy: float = 1.0,   # Награда за победу в бою против каждого отряда
        reward_ruin_clear_bonus: Optional[float] = None,  # Небольшой бонус сверху за зачистку руин
        reward_all_enemies: float = 10.0,   # База: 1-й целевой город = x1, 2-й = x5
        reward_loss: float = -5.0,          # Штраф за поражение кампании
        reward_timeout: float = -3.0,       # Штраф за таймаут (лимит шагов)
        reward_turn_penalty: float = 0.02,  # Штраф за завершение хода (REST)
        reward_needaunit_turn_penalty: float = 0.05,  # Штраф за ход, пока герою нужен найм.
        reward_repeat_position_penalty: float = 0.015,  # Штраф за повторное посещение клетки.
        reward_repeat_position_penalty_cap: float = 0.08,  # Кеп на повторный штраф за 1 шаг.
        reward_backtrack_penalty: float = 0.03,  # Штраф за микро-цикл A->B->A.
        reward_no_movement_penalty: float = 0.02,  # Штраф если движение не сменило позицию.
        exp_reward_norm_k: float = 100.0,   # Нормализация опыта: norm=exp/(exp+k)
        reward_exp_weight: float = 0.2,     # Вес exp-компоненты
        reward_survival_alive_weight: float = 0.5,  # Вес доли выживших BLUE
        reward_survival_hp_weight: float = 0.5,     # Вес средней доли HP BLUE
        reward_unit_upgrade: float = 1.0,
        reward_unit_tier_bonus: float = 0.0,
        reward_unit_tier3_multiplier: float = 1.0,
        reward_enemies_defeated_weight: float = 0.0,
        battle_reward_scale: float = 0.25,
        battle_reward_win: float = 2.0,
        battle_reward_loss: float = -1.0,
        battle_reward_step: float = 0.01,
        reward_battle_item_equip: float = 0.02,
        reward_battle_item_use: float = 0.05,
        reward_combat_potion_battle_participation: float = 0.05,
        reward_sell_junk_item: float = 0.05,
        reward_unit_swap_penalty: float = 0.004,
        reward_spell_learn: float = 2.0,
        reward_spell_cast: float = 0.005,
        reward_summon_hero_battle_engage: float = 0.05,
        reward_castle_heal_per_hp: float = 0.01,
        reward_castle_revive: float = 0.5,
        persist_blue_hp: bool = True,
        log_enabled: bool = False,
        detailed_step_info: bool = True,
        freeze_dynamic_action_layout: bool = False,
        scripted_capital_bot_enabled: bool = True,
        empire_territory_enabled: bool = True,
        max_grid_steps: int = 1800,
        Realcapital: int = 1,
        typeoflord: int = 1,
        campaign_objective: str = "cities",
        reward_objective_city_reached: float = 6.0,
        reward_green_dragon_reached: float = 6.0,
        reward_green_dragon_objective_multiplier: float = 2.0,
        reward_dragon_damage_fraction: float = 0.0,
        reward_dragon_rounds_record: float = 0.0,
        reward_no_dragon_victory_late_step_penalty: float = 0.01,
        reward_no_dragon_victory_late_step_penalty_cap: float = 0.05,
        reward_no_dragon_victory_late_step_start_fraction: float = 0.35,
    ):
        """Собрать конфигурацию кампании и базовое состояние среды.

        Конструктор нормализует параметры наград, выбранную столицу и тип лорда,
        подготавливает статическую карту сценария, торговые/магические/наёмные
        точки, кеши инвентаря, динамическую раскладку действий и Gym spaces.
        Полное эпизодное состояние пересоздаётся в reset(), но эти атрибуты
        нужны миксинам, observation и action mask ещё до первого сброса.
        """
        super().__init__()

        self.grid_size = grid_size
        self.reward_engage_battle = reward_engage_battle  # Награда за участие в битве
        self.reward_defeat_enemy = reward_defeat_enemy    # Награда за победу в битве
        if reward_ruin_clear_bonus is None:
            reward_ruin_clear_bonus = max(0.1, float(reward_defeat_enemy) * 0.25)
        self.reward_ruin_clear_bonus = max(0.0, float(reward_ruin_clear_bonus))
        self.reward_all_enemies = reward_all_enemies      # База награды за захват целевого города
        self.reward_loss = reward_loss
        self.reward_timeout = reward_timeout
        self.reward_turn_penalty = reward_turn_penalty
        self.reward_needaunit_turn_penalty = max(0.0, float(reward_needaunit_turn_penalty))
        self.reward_repeat_position_penalty = max(0.0, float(reward_repeat_position_penalty))
        self.reward_repeat_position_penalty_cap = max(0.0, float(reward_repeat_position_penalty_cap))
        self.reward_backtrack_penalty = max(0.0, float(reward_backtrack_penalty))
        self.reward_no_movement_penalty = max(0.0, float(reward_no_movement_penalty))
        self.exp_reward_norm_k = max(1.0, float(exp_reward_norm_k))
        self.reward_exp_weight = max(0.0, float(reward_exp_weight))
        self.reward_survival_alive_weight = max(0.0, float(reward_survival_alive_weight))
        self.reward_survival_hp_weight = max(0.0, float(reward_survival_hp_weight))
        self.reward_unit_upgrade = max(0.0, float(reward_unit_upgrade))
        self.reward_unit_tier_bonus = max(0.0, float(reward_unit_tier_bonus))
        self.reward_unit_tier3_multiplier = max(0.0, float(reward_unit_tier3_multiplier))
        self.reward_enemies_defeated_weight = max(0.0, float(reward_enemies_defeated_weight))
        self.battle_reward_scale = max(0.0, float(battle_reward_scale))
        self.battle_reward_win = float(battle_reward_win)
        self.battle_reward_loss = float(battle_reward_loss)
        self.battle_reward_step = float(battle_reward_step)
        self.reward_battle_item_equip = max(0.0, float(reward_battle_item_equip))
        self.reward_battle_item_use = max(0.0, float(reward_battle_item_use))
        self.reward_combat_potion_battle_participation = max(
            0.0,
            float(reward_combat_potion_battle_participation),
        )
        self.reward_sell_junk_item = max(0.0, float(reward_sell_junk_item))
        self.reward_unit_swap_penalty = max(0.0, float(reward_unit_swap_penalty))
        self.reward_spell_learn = max(0.0, float(reward_spell_learn))
        self.reward_spell_cast = max(0.0, float(reward_spell_cast))
        self.reward_summon_hero_battle_engage = max(
            0.0,
            float(reward_summon_hero_battle_engage),
        )
        self.reward_castle_heal_per_hp = max(0.0, float(reward_castle_heal_per_hp))
        self.reward_castle_revive = max(0.0, float(reward_castle_revive))
        self.persist_blue_hp = persist_blue_hp
        self.log_enabled = log_enabled
        self.detailed_step_info = bool(detailed_step_info)
        self.freeze_dynamic_action_layout = bool(freeze_dynamic_action_layout)
        self.scripted_capital_bot_config_enabled = bool(scripted_capital_bot_enabled)
        self.empire_territory_enabled = bool(empire_territory_enabled)
        self.max_grid_steps = max(1, int(max_grid_steps))
        self.campaign_objective = self._normalize_campaign_objective(campaign_objective)
        self.reward_objective_city_reached = max(0.0, float(reward_objective_city_reached))
        self.reward_dragon_damage_fraction = max(0.0, float(reward_dragon_damage_fraction))
        self.reward_dragon_rounds_record = max(0.0, float(reward_dragon_rounds_record))
        self.reward_green_dragon_reached = max(0.0, float(reward_green_dragon_reached))
        self.reward_green_dragon_objective_multiplier = max(
            0.0,
            float(reward_green_dragon_objective_multiplier),
        )
        self.reward_no_dragon_victory_late_step_penalty = max(
            0.0,
            float(reward_no_dragon_victory_late_step_penalty),
        )
        self.reward_no_dragon_victory_late_step_penalty_cap = max(
            0.0,
            float(reward_no_dragon_victory_late_step_penalty_cap),
        )
        self.reward_no_dragon_victory_late_step_start_fraction = min(
            1.0,
            max(0.0, float(reward_no_dragon_victory_late_step_start_fraction)),
        )
        # Базовый лимит шагов внутри одного хода кампании.
        self.moves_per_turn = int(self.MOVES_PER_TURN)
        self.turn_norm_k = max(
            1.0,
            float(self.max_grid_steps) / float(self.GRID_STEPS_PER_TURN),
        )
        self.gold_norm_k = max(1.0, self.turn_norm_k * float(self.GOLD_PER_TURN))
        # Realcapital: 1 = empire, 2 = legions, 3 = mountain_clans, 4 = undead_hordes, 5 = elves
        self.Realcapital = self._normalize_capital_id(Realcapital)
        # typeoflord: 1 = warrior, 2 = mage, 3 = archer/thief.
        # Значения typeoflord для всех фракций трактуются так:
        #   1 = повелитель воинов
        #   2 = повелитель магов
        #   3 = повелитель воров
        self.typeoflord = self._normalize_typeoflord(typeoflord)
        self._reset_buildings_state()
        self._reset_spells_state()

        enemy_positions = scale_enemy_positions(self.grid_size)
        self.castle_heal_tiles: Tuple[Tuple[int, int], ...] = resolve_heal_tiles(
            self.grid_size,
            enemy_positions,
            start_position=self.CASTLE_POS,
            heal_tile_count=self.CASTLE_HEAL_TILE_COUNT,
        )
        obstacle_tiles = resolve_obstacle_tiles(
            self.grid_size,
            enemy_positions,
            heal_tiles=self.castle_heal_tiles,
            start_position=self.CASTLE_POS,
        )
        chest_contents = resolve_chests(
            self.grid_size,
            enemy_positions,
            heal_tiles=self.castle_heal_tiles,
            obstacle_tiles=obstacle_tiles,
            start_position=self.CASTLE_POS,
        )
        mana_reserved_tiles = set(enemy_positions.values())
        mana_reserved_tiles.update(self.castle_heal_tiles)
        mana_reserved_tiles.update(obstacle_tiles)
        mana_reserved_tiles.update(chest_contents.keys())
        mana_reserved_tiles.add(self.CASTLE_POS)
        mana_sources = resolve_mana_sources(
            self.grid_size,
            reserved_tiles=mana_reserved_tiles,
        )
        self._static_enemy_positions = dict(enemy_positions)
        self._static_castle_heal_tiles: Tuple[Tuple[int, int], ...] = tuple(self.castle_heal_tiles)
        self._static_obstacle_tiles: Tuple[Tuple[int, int], ...] = tuple(obstacle_tiles)
        self._static_chests: Dict[Tuple[int, int], Tuple[str, ...]] = dict(chest_contents)
        self._static_mana_sources: Dict[Tuple[int, int], Dict[str, object]] = {
            tuple(pos): dict(meta)
            for pos, meta in mana_sources.items()
        }
        self._static_legions_settlement_territory_sources: Tuple[Dict[str, object], ...] = (
            resolve_legions_settlement_territory_sources(
                self.grid_size,
                enemy_positions,
            )
        )
        self.legions_settlement_territory_source_by_name: Dict[str, Dict[str, object]] = {
            str(source_data.get("name", "") or ""): dict(source_data)
            for source_data in self._static_legions_settlement_territory_sources
        }
        self.legions_settlement_source_tile_by_name: Dict[str, Tuple[int, int]] = {
            str(source_data.get("name", "") or ""): tuple(source_data.get("source_tile", (0, 0)))
            for source_data in self._static_legions_settlement_territory_sources
        }
        self.legions_settlement_source_name_by_tile: Dict[Tuple[int, int], str] = {
            tuple(source_data.get("source_tile", ())): str(source_data.get("name", "") or "")
            for source_data in self._static_legions_settlement_territory_sources
            if len(tuple(source_data.get("source_tile", ()))) == 2
        }
        self.legions_settlement_territory_source_name_by_enemy_id: Dict[int, str] = {
            int(enemy_id): str(source_data.get("name", "") or "")
            for source_data in self._static_legions_settlement_territory_sources
            for enemy_id in tuple(source_data.get("required_enemy_ids", ()))
        }
        self._init_merchant_site_metadata()
        self._init_spell_shop_site_metadata()
        self._init_mercenary_site_metadata()
        self._init_trainer_site_metadata()

        # Подсреда grid: враги и препятствия автоматически масштабируются под размер карты.
        self.grid_env = GridWorldEnv(
            grid_size=grid_size,
            enemy_positions=self._static_enemy_positions,
            obstacle_positions=self._static_obstacle_tiles,
            start_position=self.CASTLE_POS,
            max_steps=max_grid_steps,
        )
        base_water_tiles = _load_campaign_base_water_tiles()
        self.water_tiles: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_water_tiles)
            )
        ) if base_water_tiles else ()
        base_forest_tiles = _load_campaign_base_forest_tiles()
        self.forest_tiles: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_forest_tiles)
            )
        ) if base_forest_tiles else ()
        base_road_tiles = _load_campaign_base_road_tiles()
        self.road_tiles: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_road_tiles)
            )
        ) if base_road_tiles else ()
        self._sync_grid_terrain_positions()
        self.legions_territory_source_tile: Tuple[int, int] = tuple(
            int(coord) for coord in self.grid_env.start_position
        )
        base_gold_mine_tiles = _load_legions_territory_base_gold_mine_tiles()
        self.gold_mine_tiles: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                tuple(tile)
                for tile in scale_static_tiles(self.grid_size, base_gold_mine_tiles)
            )
        ) if base_gold_mine_tiles else ()
        empire_source_tile = self._static_enemy_positions.get(
            int(self.EMPIRE_TERRITORY_SOURCE_ENEMY_ID),
            scale_static_tiles(self.grid_size, (tuple(self.EMPIRE_TERRITORY_SOURCE_TILE),))[0],
        )
        self.empire_territory_source_tile: Tuple[int, int] = tuple(
            int(coord) for coord in empire_source_tile
        )
        self.legions_territory_forbidden_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_forbidden_tiles()
        )
        self.legions_territory_forbidden_tile_set: set[Tuple[int, int]] = set(
            self.legions_territory_forbidden_tiles
        )
        self.legions_territory_path_blocked_tiles: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_path_blocked_tiles()
        )
        self.legions_territory_path_blocked_tile_set: set[Tuple[int, int]] = set(
            self.legions_territory_path_blocked_tiles
        )
        self.legions_territory_path_distances: Dict[Tuple[int, int], int] = (
            self._build_legions_territory_path_distances()
        )
        self.legions_territory_order: Tuple[Tuple[int, int], ...] = (
            self._build_legions_territory_order()
        )
        self.legions_settlement_territory_orders_by_name: Dict[str, Tuple[Tuple[int, int], ...]] = (
            self._build_legions_settlement_territory_orders_by_name()
        )
        self.legions_territory_tiles: Tuple[Tuple[int, int], ...] = ()
        self.legions_territory_tile_set: set[Tuple[int, int]] = set()
        self._reset_legions_settlement_territory_state()
        self.legions_captured_gold_mine_tiles: Tuple[Tuple[int, int], ...] = ()
        self.legions_captured_gold_mine_count: int = 0
        if self.empire_territory_enabled:
            self.empire_territory_forbidden_tiles: Tuple[Tuple[int, int], ...] = (
                self._build_empire_territory_forbidden_tiles()
            )
            self.empire_territory_forbidden_tile_set: set[Tuple[int, int]] = set(
                self.empire_territory_forbidden_tiles
            )
            self.empire_territory_path_blocked_tiles: Tuple[Tuple[int, int], ...] = (
                self._build_empire_territory_path_blocked_tiles()
            )
            self.empire_territory_path_blocked_tile_set: set[Tuple[int, int]] = set(
                self.empire_territory_path_blocked_tiles
            )
            self.empire_territory_path_distances: Dict[Tuple[int, int], int] = (
                self._build_empire_territory_path_distances()
            )
            self.empire_territory_order: Tuple[Tuple[int, int], ...] = (
                self._build_empire_territory_order()
            )
        else:
            self.empire_territory_forbidden_tiles = ()
            self.empire_territory_forbidden_tile_set = set()
            self.empire_territory_path_blocked_tiles = ()
            self.empire_territory_path_blocked_tile_set = set()
            self.empire_territory_path_distances = {}
            self.empire_territory_order = ()
        self.empire_territory_tiles: Tuple[Tuple[int, int], ...] = ()
        self.empire_territory_tile_set: set[Tuple[int, int]] = set()
        self._legions_territory_grid_obs_cache = np.zeros(0, dtype=np.float32)
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        self._ensure_map_layout_consistency()
        self.battle_env: Optional[BattleEnv] = None
        self.grid_obs_base_size = int(self.grid_env.observation_space.shape[0])
        self._init_enemy_obs_metadata()

        # Текущий режим
        self.mode = self.MODE_GRID
        self.current_enemy_id: Optional[int] = None
        self.battle_origin_pos: Optional[Tuple[int, int]] = None
        self.current_battle_context: Dict[str, object] = {}

        # Сохраняем состояние BLUE команды между боями
        self.blue_team_state: Optional[List[Dict]] = None
        self.heal_bottles_used: int = 0
        self.healing_bottles_used: int = 0
        self.revive_bottles_used: int = 0
        self.turns: int = 0
        self.campaign_steps: int = 0
        self.gold: float = 0.0
        self.moves: int = self.moves_per_turn
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self.active_spells = self._get_spells_for_capital(self.Realcapital)
        self.spell_keys = self._get_spell_keys(self.active_spells)
        self.active_hire_options: List[Dict[str, object]] = []
        self.active_mercenary_hire_options: List[Dict[str, object]] = []
        self.settlement_upgrade_names = list(self.legions_settlement_territory_source_by_name.keys())
        self._refresh_dynamic_action_layout()
        self._dynamic_action_layout_capital = int(self.Realcapital)
        self._dynamic_action_layout_typeoflord = int(self.typeoflord)
        self.chests: Dict[Tuple[int, int], Tuple[str, ...]] = dict(self._static_chests)
        self.mana_sources: Dict[Tuple[int, int], Dict[str, object]] = {
            tuple(pos): dict(meta)
            for pos, meta in self._static_mana_sources.items()
        }
        self._reset_mana_state()
        self.spell_learning_locked = False
        self._inventory_version: int = 0
        self._inventory_cache_version: int = -1
        self.inventory_dirty: bool = True
        self.equipment_dirty: bool = True
        self.battle_items_dirty: bool = True
        self.boots_dirty: bool = True
        self._equipment_dirty: bool = True
        self._equipment_refresh_signature: Optional[Tuple[object, ...]] = None
        self._battle_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._equipped_hero_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._equipped_artifact_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._equipped_banner_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._equipped_book_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._equipped_boot_items_sync_signature: Optional[Tuple[object, ...]] = None
        self._battle_equip_action_mask_signature: Optional[Tuple[object, ...]] = None
        self._battle_equip_action_mask_cache: Tuple[bool, ...] = ()
        self._merchant_autosale_checked_inventory_version: Optional[int] = None
        self._merchant_autosale_checked_position: Optional[Tuple[int, int]] = None
        self.heroitems: List[object] = []
        self._inventory_cache_valid: bool = False
        self._inventory_cache_signature: Tuple[str, ...] = ()
        self._hero_item_counts_cache: Dict[str, int] = {}
        self._scroll_item_counts_cache: Dict[str, int] = {}
        self._scroll_inventory_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._available_scroll_spell_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._scroll_cast_slot_entries_cache: Tuple[Dict[str, object], ...] = ()
        self._total_scroll_count_cache: int = 0
        self.equipped_hero_items: List[Optional[str]] = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_hero_item_uses_left: List[Optional[int]] = [None] * int(
            self.BATTLE_EQUIP_SLOTS
        )
        self.equipped_hero_item_uses_left_item_names: List[Optional[str]] = [None] * int(
            self.BATTLE_EQUIP_SLOTS
        )
        self.equipped_artifact_items: List[Optional[str]] = [None] * int(
            self.ARTIFACT_EQUIP_SLOTS
        )
        self.equipped_banner_items: List[Optional[str]] = [None] * int(
            self.BANNER_EQUIP_SLOTS
        )
        self.equipped_book_items: List[Optional[str]] = [None] * int(
            self.BOOK_EQUIP_SLOTS
        )
        self.equipped_boot_items: List[Optional[str]] = [None] * int(
            self.BOOTS_EQUIP_SLOTS
        )
        self.artifact_auto_equip_next_slot: int = 0
        self.battle_item_equip_used_this_turn: bool = False
        self.book_equip_used_this_turn: bool = False
        self.enemy_team_states: Dict[int, List[Dict]] = self._create_initial_enemy_team_states()
        self.enemy_map_spell_effects: Dict[int, set[str]] = {}
        self.blue_map_spell_effects: set[str] = set()
        self.spell_cast_counts_by_id_this_turn: Dict[str, int] = {}
        self.summon_hero_battle_bonus_pending: bool = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn: set[int] = set()
        self.extra_heal_bottles: int = 0
        self.extra_healing_bottles: int = 0
        self.extra_revive_bottles: int = 0
        self.merchant_stocks: Dict[str, Dict[str, int]] = self._create_initial_merchant_stocks()
        self.spell_shop_stocks: Dict[str, Dict[str, int]] = self._create_initial_spell_shop_stocks()
        self.spell_shop_purchased_spell_ids: set[str] = set()
        self.scroll_magic_unlocked: bool = False
        self.mercenary_site_rosters: Dict[str, List[Dict[str, object]]] = (
            self._create_initial_mercenary_site_rosters()
        )
        self.active_invulnerability_potion_positions: set[int] = set()
        self.active_strength_potion_positions: set[int] = set()
        self.active_energy_elixir_positions: set[int] = set()
        self.active_haste_elixir_positions: set[int] = set()
        self.active_fire_ward_positions: set[int] = set()
        self.active_earth_ward_positions: set[int] = set()
        self.active_water_ward_positions: set[int] = set()
        self.active_air_ward_positions: set[int] = set()
        self.active_potion_effect_positions: Dict[str, set[int]] = {}
        self.battle_items_equipped_total: int = 0
        self.battle_items_used_total: int = 0
        self.books_equipped_total: int = 0
        self.grid_unit_swap_actions_used_this_turn: set[int] = set()
        self.grid_unit_swaps_this_turn: int = 0
        self.grid_unit_swaps_total: int = 0
        self.captured_objective_cities: set[str] = set()
        self.objective_city_reached_reward_granted: set[str] = set()
        self._objective_dragon_min_hp_this_episode: float | None = None
        self._objective_dragon_best_rounds_this_episode: int = 0
        self.cleared_ruin_enemy_ids: set[int] = set()
        self.last_ruin_reward: Optional[Dict[str, object]] = None
        self._init_scripted_capital_bot_state()
        self._refresh_faction_territories()
        self._sync_grid_chest_positions()
        self._sync_grid_mana_sources()
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        self._init_campaign_grid_obs_metadata()
        self._refresh_legions_territory_grid_obs_cache()
        self.GRID_OBS_SIZE = (
            self.grid_obs_base_size
            + self.grid_enemy_obs_size
            + self.grid_campaign_obs_size
        )
        self._refresh_grid_obs_slices()
        self._refresh_full_obs_slices()
        total_actions = (
            self.GRID_SWAP_UNIT_ACTION_START
            + int(self.GRID_SWAP_UNIT_ACTION_COUNT)
        )
        # Action space: grid is wider than battle (battle = 39 actions).
        self.action_space = spaces.Discrete(total_actions)

        # Observation space: режим + grid + battle + ход + золото
        total_obs = 1 + self.GRID_OBS_SIZE + self.BATTLE_OBS_SIZE + 2
        self.observation_space = spaces.Box(
            low=np.zeros(total_obs, dtype=np.float32),
            high=np.ones(total_obs, dtype=np.float32),
            shape=(total_obs,),
            dtype=np.float32,
        )

        # Логи
        self._campaign_logs: List[str] = []
        self.grid_visit_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
    def _starting_blue_roster_spec(self) -> Optional[Dict[int, str]]:
        """Подобрать стартовый состав BLUE для текущей фракции и типа лорда.

        Ключ результата — позиция в боевой сетке 7-12, значение — имя юнита из
        таблиц данных. Если для пары Realcapital/typeoflord нет отдельной
        раскладки, вызывающий код откатывается к базовому UNITS_BLUE.
        """
        lord_type = self._normalize_typeoflord(getattr(self, "typeoflord", 1))
        Realcapital = self._normalize_capital_id(getattr(self, "Realcapital", 1))
        roster_specs: Dict[Tuple[int, int], Dict[int, str]] = {
            (1, 1): {
                7: "\u0421\u043a\u0432\u0430\u0435\u0440",
                8: "\u0420\u044b\u0446\u0430\u0440\u044c \u043d\u0430 \u043f\u0435\u0433\u0430\u0441\u0435",
                9: "\u0421\u043a\u0432\u0430\u0435\u0440",
                11: "\u0421\u043b\u0443\u0436\u043a\u0430",
            },
            (1, 2): {
                7: "\u0421\u043a\u0432\u0430\u0435\u0440",
                9: "\u0421\u043a\u0432\u0430\u0435\u0440",
                10: "\u0410\u0440\u0445\u0438\u043c\u0430\u0433",
                12: "\u0421\u043b\u0443\u0436\u043a\u0430",
            },
            (1, 3): {
                7: "\u0421\u043a\u0432\u0430\u0435\u0440",
                9: "\u0421\u043a\u0432\u0430\u0435\u0440",
                10: "\u0420\u0435\u0439\u043d\u0434\u0436\u0435\u0440 \u043b\u044e\u0434\u0435\u0439",
                12: "\u0421\u043b\u0443\u0436\u043a\u0430",
            },
            (2, 2): {
                7: "\u041e\u0434\u0435\u0440\u0436\u0438\u043c\u044b\u0439",
                9: "\u041e\u0434\u0435\u0440\u0436\u0438\u043c\u044b\u0439",
                10: "\u0410\u0440\u0445\u0438\u0434\u044c\u044f\u0432\u043e\u043b",
                12: "\u0421\u0435\u043a\u0442\u0430\u043d\u0442",
            },
            (2, 3): {
                7: "\u041e\u0434\u0435\u0440\u0436\u0438\u043c\u044b\u0439",
                9: "\u041e\u0434\u0435\u0440\u0436\u0438\u043c\u044b\u0439",
                10: "\u0421\u043e\u0432\u0435\u0442\u043d\u0438\u043a",
                12: "\u0421\u0435\u043a\u0442\u0430\u043d\u0442",
            },
            (3, 1): {
                7: "\u0413\u043d\u043e\u043c",
                8: "\u041a\u043e\u0440\u043e\u043b\u0435\u0432\u0441\u043a\u0438\u0439 \u0441\u0442\u0440\u0430\u0436",
                9: "\u0413\u043d\u043e\u043c",
                11: "\u0422\u0440\u0430\u0432\u043d\u0438\u0446\u0430",
            },
            (3, 2): {
                7: "\u0413\u043d\u043e\u043c",
                9: "\u0413\u043d\u043e\u043c",
                10: "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u0437\u043d\u0430\u043d\u0438\u0439",
                12: "\u0422\u0440\u0430\u0432\u043d\u0438\u0446\u0430",
            },
            (3, 3): {
                7: "\u0413\u043d\u043e\u043c",
                9: "\u0413\u043d\u043e\u043c",
                10: "\u0418\u043d\u0436\u0435\u043d\u0435\u0440",
                12: "\u0422\u0440\u0430\u0432\u043d\u0438\u0446\u0430",
            },
            (4, 1): {
                7: "\u0412\u043e\u0438\u043d",
                8: "\u0420\u044b\u0446\u0430\u0440\u044c \u0441\u043c\u0435\u0440\u0442\u0438",
                9: "\u0412\u043e\u0438\u043d",
                11: "\u0410\u0434\u0435\u043f\u0442",
            },
            (4, 2): {
                7: "\u0412\u043e\u0438\u043d",
                9: "\u0412\u043e\u0438\u043d",
                10: "\u041a\u043e\u0440\u043e\u043b\u0435\u0432\u0430 \u043b\u0438\u0447",
                12: "\u0410\u0434\u0435\u043f\u0442",
            },
            (4, 3): {
                7: "\u0412\u043e\u0438\u043d",
                9: "\u0412\u043e\u0438\u043d",
                10: "\u041d\u043e\u0441\u0444\u0435\u0440\u0430\u0442\u0443",
                12: "\u0410\u0434\u0435\u043f\u0442",
            },
            (5, 1): {
                7: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                8: "\u041b\u0435\u0441\u043d\u043e\u0439 \u043b\u043e\u0440\u0434",
                9: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                11: "\u041c\u0435\u0434\u0438\u0443\u043c",
            },
            (5, 2): {
                7: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                9: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                10: "\u0414\u0440\u0438\u0430\u0434\u0430",
                12: "\u041c\u0435\u0434\u0438\u0443\u043c",
            },
            (5, 3): {
                7: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                9: "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a",
                10: "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u043b\u0435\u0441\u0430",
                12: "\u041c\u0435\u0434\u0438\u0443\u043c",
            },
        }
        return roster_specs.get((Realcapital, lord_type))

    def _create_starting_blue_team(self) -> List[Dict]:
        """Создать стартовый BLUE-отряд как список боевых unit dict.

        Метод переводит сценарную раскладку из имён юнитов в реальные структуры
        BattleEnv, добавляет пустые placeholder-слоты для незанятых позиций и
        сохраняет порядок GRID_BOTTLE_POSITIONS. При отсутствии специальной
        раскладки используется глубокая копия классического UNITS_BLUE.
        """
        spec = self._starting_blue_roster_spec()
        if spec is None:
            return deepcopy(UNITS_BLUE)

        team: List[Dict] = []
        for position in self.GRID_BOTTLE_POSITIONS:
            unit_name = spec.get(int(position))
            if not unit_name:
                team.append(placeholder_unit("blue", int(position)))
                continue
            unit_data = self._find_unit_data_by_name(unit_name)
            if unit_data is None:
                raise ValueError(f"Unknown starting unit {unit_name!r}")
            team.append(self._build_unit_from_data(unit_data, "blue", int(position)))
        return team

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Начать новый эпизод кампании и вернуть первое наблюдение.

        Reset применяет optional Realcapital/typeoflord, пересобирает здания,
        заклинания и динамический action layout, очищает бой, инвентарь, руины,
        торговые запасы, ману, эффекты зелий, территории и scripted-bot state.
        Затем сбрасывает вложенный GridEnv, синхронизирует карту с текущими
        сундуками/источниками/сайтами и формирует стартовый info для отладки.
        """
        super().reset(seed=seed)
        requested_capital = (
            self._normalize_capital_id(options.get("Realcapital"))
            if isinstance(options, dict) and "Realcapital" in options
            else self._normalize_capital_id(getattr(self, "Realcapital", 1))
        )
        requested_typeoflord = (
            self._normalize_typeoflord(options.get("typeoflord"))
            if isinstance(options, dict) and "typeoflord" in options
            else self._normalize_typeoflord(getattr(self, "typeoflord", 1))
        )
        if bool(getattr(self, "freeze_dynamic_action_layout", False)):
            layout_capital = int(getattr(self, "_dynamic_action_layout_capital", requested_capital))
            layout_typeoflord = int(
                getattr(self, "_dynamic_action_layout_typeoflord", requested_typeoflord)
            )
            if requested_capital != layout_capital or requested_typeoflord != layout_typeoflord:
                raise ValueError(
                    "Cannot change Realcapital/typeoflord while freeze_dynamic_action_layout=True"
                )
            self.Realcapital = layout_capital
            self.typeoflord = layout_typeoflord
        else:
            self.Realcapital = requested_capital
            self.typeoflord = requested_typeoflord
        if isinstance(options, dict) and "campaign_objective" in options:
            self.campaign_objective = self._normalize_campaign_objective(
                options.get("campaign_objective")
            )
        if isinstance(options, dict) and "scripted_capital_bot_enabled" in options:
            self.scripted_capital_bot_config_enabled = bool(
                options.get("scripted_capital_bot_enabled")
            )
            self.scripted_capital_bot_enabled = bool(self.scripted_capital_bot_config_enabled)

        self._reset_buildings_state()
        self.active_buildings = self._get_buildings_for_capital(self.Realcapital)
        self.building_keys = self._get_building_keys(self.active_buildings)
        self._reset_spells_state()
        self.active_spells = self._get_spells_for_capital(self.Realcapital)
        self.spell_keys = self._get_spell_keys(self.active_spells)
        self.settlement_upgrade_names = list(self.legions_settlement_territory_source_by_name.keys())
        if not bool(getattr(self, "freeze_dynamic_action_layout", False)):
            self._refresh_dynamic_action_layout()
            self._dynamic_action_layout_capital = int(self.Realcapital)
            self._dynamic_action_layout_typeoflord = int(self.typeoflord)
        self.mode = self.MODE_GRID
        self.current_enemy_id = None
        self.battle_origin_pos = None
        self.current_battle_context = {}
        # Стартовое состояние BLUE — копия базовых юнитов, чтобы
        # вне боя можно было применять лечение до первого боя.
        self.blue_team_state = self._create_starting_blue_team()
        self._sync_hero_progression_flags(self.blue_team_state)
        self.heal_bottles_used = 0
        self.healing_bottles_used = 0
        self.revive_bottles_used = 0
        self.turns = 0
        self.campaign_steps = 0
        self.gold = 0.0
        self.spell_learning_locked = False
        self.battle_env = None
        self._campaign_logs = []
        self._inventory_version = int(getattr(self, "_inventory_version", 0) or 0)
        self._inventory_cache_version = -1
        self.inventory_dirty = True
        self.equipment_dirty = True
        self.battle_items_dirty = True
        self.boots_dirty = True
        self._equipment_dirty = True
        self._equipment_refresh_signature = None
        self._battle_items_sync_signature = None
        self._equipped_hero_items_sync_signature = None
        self._equipped_artifact_items_sync_signature = None
        self._equipped_banner_items_sync_signature = None
        self._equipped_book_items_sync_signature = None
        self._equipped_boot_items_sync_signature = None
        self._battle_equip_action_mask_signature = None
        self._battle_equip_action_mask_cache = ()
        self._merchant_autosale_checked_inventory_version = None
        self._merchant_autosale_checked_position = None
        self.heroitems = []
        self._invalidate_inventory_cache()
        self.equipped_hero_items = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_hero_item_uses_left = [None] * int(self.BATTLE_EQUIP_SLOTS)
        self.equipped_hero_item_uses_left_item_names = [None] * int(
            self.BATTLE_EQUIP_SLOTS
        )
        self.equipped_artifact_items = [None] * int(self.ARTIFACT_EQUIP_SLOTS)
        self.equipped_banner_items = [None] * int(self.BANNER_EQUIP_SLOTS)
        self.equipped_book_items = [None] * int(self.BOOK_EQUIP_SLOTS)
        self.equipped_boot_items = [None] * int(self.BOOTS_EQUIP_SLOTS)
        self.artifact_auto_equip_next_slot = 0
        self._sync_moves_per_turn_with_hero(units=self.blue_team_state, refill=True)
        if bool(getattr(self, "freeze_dynamic_action_layout", False)):
            self._reset_blue_unit_swap_counters()
        else:
            self._reset_blue_unit_swap_state()
        self.battle_item_equip_used_this_turn = False
        self.book_equip_used_this_turn = False
        self.enemy_team_states = self._create_initial_enemy_team_states()
        self.enemy_map_spell_effects = {}
        self.blue_map_spell_effects = set()
        self.spell_cast_counts_by_id_this_turn = {}
        self.summon_hero_battle_bonus_pending = False
        self.summon_hero_battle_bonus_enemy_ids_this_turn = set()
        self.extra_heal_bottles = 0
        self.extra_healing_bottles = 0
        self.extra_revive_bottles = 0
        self._sync_counter_backed_battle_items()
        self.merchant_stocks = self._create_initial_merchant_stocks()
        self.spell_shop_stocks = self._create_initial_spell_shop_stocks()
        self.spell_shop_purchased_spell_ids = set()
        self.scroll_magic_unlocked = False
        self.mercenary_site_rosters = self._create_initial_mercenary_site_rosters()
        self.active_invulnerability_potion_positions = set()
        self.active_strength_potion_positions = set()
        self.active_energy_elixir_positions = set()
        self.active_haste_elixir_positions = set()
        self.active_fire_ward_positions = set()
        self.active_earth_ward_positions = set()
        self.active_water_ward_positions = set()
        self.active_air_ward_positions = set()
        self.active_potion_effect_positions = {}
        self.battle_items_equipped_total = 0
        self.battle_items_used_total = 0
        self.books_equipped_total = 0
        self.combat_potion_battle_bonus_pending = False
        self.green_dragon_reached_reward_granted = False
        self.captured_objective_cities = set()
        self.objective_city_reached_reward_granted = set()
        self._objective_dragon_min_hp_this_episode = None
        self._objective_dragon_best_rounds_this_episode = 0
        self.cleared_ruin_enemy_ids = set()
        self.last_ruin_reward = None
        self.chests = dict(self._static_chests)
        self.mana_sources = {
            tuple(pos): dict(meta)
            for pos, meta in self._static_mana_sources.items()
        }
        self._reset_legions_settlement_territory_state()
        self._reset_mana_state()
        self.castle_heal_tiles = self._static_castle_heal_tiles
        self._refresh_faction_territories()
        self.grid_env.enemy_positions = dict(self._static_enemy_positions)
        self.grid_env.obstacle_positions = set(self._static_obstacle_tiles)
        self._sync_grid_terrain_positions()
        self._sync_grid_chest_positions()
        self._sync_grid_mana_sources()
        self._sync_grid_merchant_positions()
        self._sync_grid_spell_shop_positions()
        self._sync_grid_mercenary_positions()
        self._sync_grid_trainer_positions()
        if not bool(getattr(self, "freeze_dynamic_action_layout", False)):
            total_actions = (
                self.GRID_SWAP_UNIT_ACTION_START
                + int(self.GRID_SWAP_UNIT_ACTION_COUNT)
            )
            self.action_space = spaces.Discrete(total_actions)

        grid_obs, grid_info = self.grid_env.reset(seed=seed)
        self._reset_scripted_capital_bot_state()
        grid_obs = self.grid_env._get_obs()
        grid_obs = self._augment_grid_obs(grid_obs)
        self._reset_stagnation_tracking(self.grid_env.agent_pos)

        self._log("=== НОВАЯ КАМПАНИЯ ===")
        self._log(f"Старт на позиции {self.grid_env.agent_pos}")
        self._log(f"Клетки лечения на карте: {self.castle_heal_tiles}")
        self._log(f"Препятствия на карте: {sorted(self.grid_env.obstacle_positions)}")
        self._log(f"Враги на карте: {list(self.grid_env.enemy_positions.items())}")
        self._log(f"Сундуки на карте: {list(self.chests.items())}")

        info = {
            "mode": "grid",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "typeoflord": int(self.typeoflord),
            "heroitems": list(self.heroitems),
            "scenario_potion_item_names": list(self.scenario_potion_item_names),
            "scenario_merchant_potion_item_names": list(self.scenario_merchant_potion_item_names),
            "scenario_battle_potion_equip_names": list(self.scenario_battle_potion_equip_names),
            "equipped_hero_items": list(self.equipped_hero_items),
            "equipped_hero_item_uses_left": list(self.equipped_hero_item_uses_left),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "books_equipped_total": int(self.books_equipped_total),
            "unit_swaps_total": int(self.grid_unit_swaps_total),
            "book_equip_used_this_turn": bool(self.book_equip_used_this_turn),
            "extra_heal_bottles": int(self.extra_heal_bottles or 0),
            "extra_healing_bottles": int(self.extra_healing_bottles or 0),
            "extra_revive_bottles": int(self.extra_revive_bottles or 0),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "active_energy_positions": sorted(self.active_energy_elixir_positions),
            "active_haste_positions": sorted(self.active_haste_elixir_positions),
            "active_fire_ward_positions": sorted(self.active_fire_ward_positions),
            "active_earth_ward_positions": sorted(self.active_earth_ward_positions),
            "active_water_ward_positions": sorted(self.active_water_ward_positions),
            "active_air_ward_positions": sorted(self.active_air_ward_positions),
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "chests_remaining": len(self.chests),
            "legions_territory_tiles": tuple(self.legions_territory_tiles),
            "legions_territory_count": len(self.legions_territory_tiles),
            "legions_captured_gold_mine_tiles": tuple(self.legions_captured_gold_mine_tiles),
            "legions_captured_gold_mine_count": int(self.legions_captured_gold_mine_count),
            "legions_gold_mine_income_per_turn": (
                int(self.legions_captured_gold_mine_count)
                * float(self.LEGIONS_GOLD_MINE_GOLD_PER_TURN)
            ),
            "empire_territory_tiles": tuple(self.empire_territory_tiles),
            "empire_territory_count": len(self.empire_territory_tiles),
        }
        info.update(self._legions_settlement_territory_info())
        info.update(self._mana_state_info())
        info.update(self._spell_state_info())
        info.update(self._ruin_progress_info())
        info.update(self._artifact_state_info())
        info.update(self._banner_state_info())
        info.update(self._book_state_info())
        info.update(self._boot_state_info())
        info.update(self._scripted_capital_bot_info())
        info.update(self._merchant_context_info(self.grid_env.agent_pos))
        info.update(self._spell_shop_context_info(self.grid_env.agent_pos))
        info.update(self._mercenary_context_info(self.grid_env.agent_pos))
        info.update(self._trainer_context_info(self.grid_env.agent_pos))

        return self._build_obs(grid_obs=grid_obs), info
    def step(self, action: int):
        """Выполнить один шаг Gym-среды в активном режиме.

        CampaignEnv имеет два режима: grid-режим обрабатывает перемещение,
        экономику и магию на карте, а battle-режим делегирует действие текущему
        BattleEnv. Action приводится к int, потому что vectorized wrappers часто
        передают numpy-скаляры или массивы из одного элемента.
        """
        # Приводим action к int (может прийти как numpy.ndarray)
        action = int(action)
        self.campaign_steps = int(getattr(self, "campaign_steps", 0) or 0) + 1

        if self.mode == self.MODE_GRID:
            return self._step_grid(action)
        else:
            return self._step_battle(action)
    def _step_grid(self, action: int):
        """Обработать действие агента на глобальной карте.

        Метод сначала маршрутизирует action в специализированные блоки:
        перестановки отряда, свитки, staff-магия, магазины, найм, здания,
        обучение заклинаний, лечение/воскрешение и зелья. Если action относится
        к движению или отдыху, он обновляет позицию GridEnv, тратит очки хода,
        собирает сундуки, проверяет столкновения, scripted-бота и условия
        победы/поражения, после чего передаёт результат в общий финализатор.
        """
        # В grid режиме:
        #   0-7 — движение, 8 — отдых,
        #   9-14 — применить подходящую банку лечения к позициям 7-12 соответственно.
        #   15-20 — применить бутыль воскрешения к позициям 7-12 соответственно.
        #   21-26 — применить зелье неуязвимости к позициям 7-12 соответственно.
        #   27-32 — применить зелье силы к позициям 7-12 соответственно.
        #   33-38 — лечение на специальных клетках карты по позициям 7-12.
        #   39-44 — воскрешение на специальных клетках карты по позициям 7-12.
        #   после 44 — найм фракционных юнитов в городе/столице;
        #   число action в этом блоке зависит от массива найма выбранной расы в hire.py.
        #   после него — отдельный блок найма в лагерях наёмников на карте.
        #   после блока наёмников — покупки у торговца, затем боевые эликсиры и постройки.
        #   после построек — изучение заклинаний активной фракции.
        #   после заклинаний — улучшение захваченных городов.
        include_detailed_info = self._include_detailed_step_info()
        if action >= self.GRID_SWAP_UNIT_ACTION_START:
            return self._step_swap_blue_unit_positions(action)
        if action >= self.grid_scroll_cast_action_start:
            return self._step_cast_scroll_spell(action)
        if action >= self.GRID_UNLOCK_SCROLL_MAGIC_ACTION:
            return self._step_unlock_scroll_magic(action)
        if action >= self.grid_staff_spell_action_start:
            return self._step_cast_staff_spell(action)
        if action >= self.grid_spell_shop_cast_action_start:
            return self._step_cast_spell_shop_spell(action)
        if action >= self.grid_map_support_spell_action_start:
            return self._step_cast_support_spell(action)
        if action >= self.grid_legion_damage_spell_action_start:
            return self._step_cast_legion_damage_spell(action)
        if action >= self.GRID_EQUIP_BOOK_ACTION_START:
            return self._step_equip_book_item(action)
        if action >= self.GRID_EQUIP_BATTLE_ITEM1_ACTION_START:
            return self._step_equip_battle_item(action)
        if action >= self.grid_settlement_upgrade_action_start:
            return self._step_upgrade_settlement(action)
        if action >= self.grid_spell_action_start:
            return self._step_learn_spell(action)
        if action >= self.GRID_BUILD_ACTION_START:
            return self._step_building(action)
        if action >= self.GRID_SPELL_SHOP_BUY_ACTION_START:
            return self._step_buy_spell_shop_spell(action)
        if action >= self.GRID_MERCHANT_POTION_BUY_ACTION_START:
            return self._step_buy_merchant_item(action)
        if action >= self.GRID_TRAINER_ACTION_START:
            return self._step_train_unit_at_trainer(action)
        if action >= self.GRID_MERCENARY_HIRE_ACTION_START:
            return self._step_hire_mercenary_unit(action)
        if action >= self.GRID_HIRE_ACTION_START:
            return self._step_hire_faction_unit(action)
        if action >= self.GRID_CASTLE_REVIVE_ACTION_START:
            return self._step_revive_in_castle(action)
        if action >= self.GRID_CASTLE_HEAL_ACTION_START:
            return self._step_heal_in_castle(action)
        if action >= self.GRID_POTION_USE_ACTION_START:
            return self._step_use_potion(action)

        grid_move_cost = self._grid_move_cost_for_action(action)

        if 0 <= action <= 7 and self.moves <= 0:
            grid_obs = self._get_grid_obs()
            info = {
                "mode": "grid",
                "agent_pos": self.grid_env.agent_pos,
                "enemies_alive": dict(self.grid_env.enemies_alive),
                "battle_triggered": False,
                "blocked_by_moves": True,
                "move_cost": int(grid_move_cost),
                "move_points_spent": 0,
                "target_terrain": self._campaign_tile_terrain(
                    self._grid_move_target_for_action(action)
                    or tuple(self.grid_env.agent_pos)
                ),
                "turns": self.turns,
                "gold": self.gold,
                "moves": self.moves,
                "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            }
            if include_detailed_info:
                info.update(
                    {
                        "heroitems": list(self.heroitems),
                        "equipped_hero_items": list(self.equipped_hero_items),
                        "equipped_hero_item_uses_left": list(
                            self.equipped_hero_item_uses_left
                        ),
                        "extra_healing_bottles": int(self.extra_healing_bottles or 0),
                        "extra_revive_bottles": int(self.extra_revive_bottles or 0),
                        "active_invulnerability_positions": sorted(
                            self.active_invulnerability_potion_positions
                        ),
                        "active_strength_positions": sorted(
                            self.active_strength_potion_positions
                        ),
                        "collected_chests": [],
                        "chests_remaining": len(self.chests),
                    }
                )
                info.update(self._merchant_context_info(self.grid_env.agent_pos))
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        grid_action = min(action, 8)

        old_pos = self.grid_env.agent_pos
        grid_obs, _grid_reward, terminated, truncated, grid_info = self.grid_env.step(
            grid_action
        )
        # Add dense navigation signal from GridWorldEnv to reduce timeout-only learning.
        reward = 0.2 * float(_grid_reward)
        grid_obs = self._augment_grid_obs(grid_obs)
        new_pos = self.grid_env.agent_pos
        stagnation_penalty = 0.0
        battle_engage_bonus = 0.0
        summon_hero_battle_bonus = 0.0
        combat_potion_battle_bonus = 0.0
        battle_move_cost = 0
        battle_move_spent = 0
        grid_move_spent = 0
        target_terrain = self._campaign_tile_terrain(new_pos)

        if old_pos != new_pos:
            self._log(f"Перемещение: {old_pos} -> {new_pos}")
        elif grid_info.get("blocked_by_obstacle"):
            self._log(
                f"Препятствие: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )
        elif grid_info.get("blocked_by_boundary"):
            self._log(
                f"Граница карты: переход {old_pos} -> {grid_info.get('blocked_target')} заблокирован"
            )

        if 0 <= grid_action <= 7:
            if not grid_info.get("blocked_by_obstacle", False):
                stagnation_penalty = self._compute_stagnation_penalty(old_pos, new_pos)
            reward += stagnation_penalty
            grid_move_spent = self._spend_grid_move_cost(grid_move_cost)

        if grid_info.get("battle_triggered"):
            enemy_id = grid_info.get("enemy_id")
            if (
                enemy_id is not None
                and self.grid_env.enemies_alive.get(enemy_id, False)
            ):
                battle_engage_bonus = float(self.reward_engage_battle)
                reward += battle_engage_bonus
                if int(enemy_id) in self.summon_hero_battle_bonus_enemy_ids_this_turn:
                    summon_hero_battle_bonus = float(self.reward_summon_hero_battle_engage)
                    reward += summon_hero_battle_bonus
                    self.summon_hero_battle_bonus_enemy_ids_this_turn.discard(int(enemy_id))
                self.summon_hero_battle_bonus_pending = bool(
                    self.summon_hero_battle_bonus_enemy_ids_this_turn
                )
            if self.combat_potion_battle_bonus_pending:
                combat_potion_battle_bonus = float(
                    self.reward_combat_potion_battle_participation
                )
                reward += combat_potion_battle_bonus
                self.combat_potion_battle_bonus_pending = False
            battle_move_cost = int(self._battle_grid_move_cost())
            battle_move_spent = int(self._apply_battle_grid_steps())
        collected_chests = self._collect_adjacent_chests()
        collected_combat_potions = self._count_collected_combat_potions(collected_chests)
        combat_potion_pickup_reward = (
            float(collected_combat_potions) * self._combat_potion_reward_value()
        )
        reward += combat_potion_pickup_reward

        info = {
            "mode": "grid",
            "agent_pos": new_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "battle_triggered": grid_info.get("battle_triggered", False),
            "rest_action": grid_info.get("rest_action", False),
            "blocked_by_obstacle": grid_info.get("blocked_by_obstacle", False),
            "blocked_by_boundary": grid_info.get("blocked_by_boundary", False),
            "blocked_obstacle_pos": grid_info.get("blocked_target"),
            "move_cost": int(grid_move_cost) if 0 <= grid_action <= 7 else 0,
            "move_points_spent": int(grid_move_spent),
            "target_terrain": str(target_terrain) if 0 <= grid_action <= 7 else "",
            "grid_reward_raw": float(_grid_reward),
            "grid_reward_scaled": float(reward),
            "stagnation_penalty": float(stagnation_penalty),
            "battle_engage_bonus": float(battle_engage_bonus),
            "summon_hero_battle_bonus": float(summon_hero_battle_bonus),
            "summon_hero_battle_bonus_applied": bool(summon_hero_battle_bonus > 0.0),
            "summon_hero_battle_bonus_pending": bool(self.summon_hero_battle_bonus_pending),
            "summon_hero_battle_bonus_enemy_ids": sorted(
                int(enemy_id) for enemy_id in self.summon_hero_battle_bonus_enemy_ids_this_turn
            ),
            "combat_potion_battle_bonus": float(combat_potion_battle_bonus),
            "battle_move_cost": int(battle_move_cost),
            "battle_move_spent": int(battle_move_spent),
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "combat_potion_battle_bonus_pending": bool(self.combat_potion_battle_bonus_pending),
            "collected_chests_count": int(len(collected_chests)),
            "collected_combat_potions": int(collected_combat_potions),
            "combat_potion_pickup_reward": float(combat_potion_pickup_reward),
        }
        if grid_info.get("battle_triggered"):
            reward = self._apply_green_dragon_reached_reward_if_needed(
                grid_info.get("enemy_id"),
                reward,
                info,
            )
            reward = self._apply_objective_city_reached_reward_if_needed(
                grid_info.get("enemy_id"),
                reward,
                info,
            )
            info["grid_reward_scaled"] = float(reward)
        if include_detailed_info:
            info.update(
                {
                    "heroitems": list(self.heroitems),
                    "extra_healing_bottles": int(self.extra_healing_bottles or 0),
                    "extra_revive_bottles": int(self.extra_revive_bottles or 0),
                    "active_invulnerability_positions": sorted(
                        self.active_invulnerability_potion_positions
                    ),
                    "active_strength_positions": sorted(
                        self.active_strength_potion_positions
                    ),
                    "collected_chests": [
                        {
                            "pos": tuple(chest_data["pos"]),
                            "items": list(chest_data["items"]),
                            "granted_items": list(chest_data.get("granted_items", [])),
                        }
                        for chest_data in collected_chests
                    ],
                    "chests_remaining": len(self.chests),
                }
            )
            info.update(self._merchant_context_info(new_pos))

        # Обработка действия REST — восстановление базового процента HP и бонусного лечения на своей столице/городе.
        if grid_info.get("rest_action"):
            pending_hire_turn_penalty = self._pending_hire_turn_penalty(
                1,
                units=self.blue_team_state,
            )
            rest_heal_percent = self._resolve_rest_heal_percent(self.grid_env.agent_pos)
            rest_heal_typeoflord_bonus_percent = self._resolve_typeoflord_rest_heal_bonus_percent()
            (
                rest_heal_bonus_percent,
                rest_heal_bonus_source,
                rest_heal_bonus_level,
            ) = self._resolve_rest_regeneration_context(self.grid_env.agent_pos)
            rest_total_percent = float(
                rest_heal_percent
                + rest_heal_typeoflord_bonus_percent
                + rest_heal_bonus_percent
            )
            healed = self._heal_blue_team(
                heal_percent=rest_heal_percent,
                bonus_percent=rest_heal_typeoflord_bonus_percent + rest_heal_bonus_percent,
            )
            if healed > 0:
                rest_heal_formula_parts = [f"{rest_heal_percent * 100.0:g}%"]
                if rest_heal_typeoflord_bonus_percent > 0.0:
                    rest_heal_formula_parts.append(
                        f"{rest_heal_typeoflord_bonus_percent * 100.0:g}%"
                    )
                if rest_heal_bonus_percent > 0.0:
                    rest_heal_formula_parts.append(f"{rest_heal_bonus_percent * 100.0:g}%")
                rest_heal_formula_text = " + ".join(rest_heal_formula_parts)
                if rest_heal_bonus_percent > 0.0:
                    if rest_heal_bonus_source == "capital":
                        self._log(
                            f"Отдых в столице: восстановлено HP у {healed} юнитов "
                            f"({rest_heal_formula_text})"
                        )
                    else:
                        self._log(
                            f"Отдых в городе {rest_heal_bonus_source} уровня {rest_heal_bonus_level}: "
                            f"восстановлено HP у {healed} юнитов "
                            f"({rest_heal_formula_text})"
                        )
                else:
                    self._log(
                        f"Отдых: восстановлено HP у {healed} юнитов "
                        f"({rest_heal_formula_text})"
                    )
            else:
                self._log("Отдых: все юниты на полном здоровье")
            # Отдых завершает ход
            reward += self._advance_turns(1)
            info["turns"] = self.turns
            info["gold"] = self.gold
            info["moves"] = self.moves
            info["healed_units"] = healed
            info["pending_hire_turn_penalty"] = float(pending_hire_turn_penalty)
            info["rest_heal_percent"] = float(rest_heal_percent)
            info["rest_heal_typeoflord_bonus_percent"] = float(
                rest_heal_typeoflord_bonus_percent
            )
            info["rest_heal_bonus_percent"] = float(rest_heal_bonus_percent)
            info["rest_heal_total_percent"] = float(rest_total_percent)
            info["rest_heal_bonus_source"] = rest_heal_bonus_source
            if rest_heal_bonus_level is not None:
                info["rest_heal_bonus_level"] = rest_heal_bonus_level

        # Проверяем столкновение с врагом
        if grid_info.get("battle_triggered"):
            enemy_id = grid_info["enemy_id"]
            self.current_enemy_id = enemy_id
            battle_origin_raw = (
                new_pos if grid_info.get("battle_triggered_by") == "adjacent" else old_pos
            )
            self.battle_origin_pos = (
                int(battle_origin_raw[0]),
                int(battle_origin_raw[1]),
            )
            self.current_battle_context = {"kind": "hero"}
            self.mode = self.MODE_BATTLE

            self._log(f"!!! СТОЛКНОВЕНИЕ С ВРАГОМ {enemy_id} !!!")
            self._log(f"Описание: {ENEMY_DESCRIPTIONS.get(enemy_id, 'Неизвестный враг')}")

            # Создаём BattleEnv с нужной RED командой
            self._init_battle(enemy_id)

            # Возвращаем battle observation
            battle_obs = self.battle_env._obs()
            info["mode"] = "battle"
            info["enemy_id"] = enemy_id

            return self._build_obs(battle_obs=battle_obs), reward, False, False, info

        # Проверяем победу (все враги побеждены)
        if self.grid_env.all_enemies_defeated():
            self._log("=== ВСЕ ВРАГИ ПОБЕЖДЕНЫ! ПОБЕДА В КАМПАНИИ! ===")
            info["campaign_result"] = "victory"
            info["campaign_victory_reason"] = "all_enemies_defeated"
            return self._finalize_grid_step_result(
                grid_obs=grid_obs,
                reward=reward,
                terminated=True,
                truncated=False,
                info=info,
            )

        # Проверяем лимит шагов на карте
        if truncated:
            self._log("=== ЛИМИТ ШАГОВ НА КАРТЕ ИСЧЕРПАН ===")
            reward += self.reward_timeout
            info["campaign_result"] = "timeout"

        return self._finalize_grid_step_result(
            grid_obs=grid_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    def _is_at_castle(self) -> bool:
        """Проверить, находится ли герой на клетке замка/лечения.

        Эти клетки открывают отдельные действия лечения и воскрешения за золото
        и используются как часть контекста отдыха на карте.
        """
        return tuple(self.grid_env.agent_pos) in self.castle_heal_tiles
    def _include_detailed_step_info(self) -> bool:
        """Определить, нужно ли собирать тяжёлый подробный info на grid step.

        В обычном обучении достаточно минимального info с наградой, позицией,
        ходами и ключевыми флагами. Подробные копии инвентаря, экипировки,
        торговых/магических контекстов и территорий включаются только для
        debug/log режимов или при явном detailed_step_info.
        """
        return bool(
            getattr(self, "detailed_step_info", True)
            or getattr(self, "debug_info", False)
            or getattr(self, "log_enabled", False)
        )
    def _finalize_grid_step_result(
        self,
        *,
        grid_obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, object],
    ):
        """Единообразно завершить любой grid-step и собрать итоговый ответ.

        Финализатор добавляет автопродажу мусорных предметов у торговца,
        обновляет награду, синхронизирует экипировку только при dirty-состоянии,
        записывает общий минимум info и строит объединённое наблюдение. Если
        включён подробный info, он дополнительно прикладывает снимки инвентаря,
        экипировки, зелий, территорий, маны, руин, магазинов и прочих контекстов.
        """
        position_raw = info.get("agent_pos", self.grid_env.agent_pos)
        if isinstance(position_raw, (list, tuple)) and len(position_raw) >= 2:
            position = (int(position_raw[0]), int(position_raw[1]))
        else:
            position = tuple(self.grid_env.agent_pos)

        finalized_info: Dict[str, object] = dict(info)
        finalized_info["mode"] = "grid"

        sale_info = self._sell_sell_only_heroitems_at_merchant(position=position)
        finalized_reward = float(reward) + float(sale_info["merchant_sale_reward"])
        finalized_info.update(sale_info)
        late_penalty = self._compute_no_dragon_victory_late_step_penalty(finalized_info)
        finalized_reward += float(late_penalty)
        finalized_info["campaign_steps"] = int(getattr(self, "campaign_steps", 0) or 0)
        if late_penalty != 0.0:
            finalized_info["no_dragon_victory_late_step_penalty"] = float(late_penalty)
        if terminated or truncated:
            finalized_reward += self._episode_enemies_defeated_bonus(finalized_info)
        equipment_signature = self._equipment_state_signature(units=self.blue_team_state)
        if (
            bool(getattr(self, "battle_items_dirty", True))
            or bool(getattr(self, "equipment_dirty", True))
            or getattr(self, "_equipment_refresh_signature", None) != equipment_signature
        ):
            self._sync_counter_backed_battle_items()
            self._sync_equipped_hero_items()
            self._refresh_campaign_equipment_effects(log=False)

        if "grid_reward_scaled" in finalized_info:
            try:
                finalized_info["grid_reward_scaled"] = (
                    float(finalized_info.get("grid_reward_scaled", 0.0))
                    + float(sale_info["merchant_sale_reward"])
                    + float(late_penalty)
                )
            except (TypeError, ValueError):
                finalized_info["grid_reward_scaled"] = float(finalized_reward)

        finalized_info["agent_pos"] = position
        finalized_info["turns"] = self.turns
        finalized_info["gold"] = self.gold
        finalized_info["moves"] = self.moves
        finalized_info["typeoflord"] = int(self.typeoflord)
        finalized_info["reward"] = float(finalized_reward)
        finalized_info["battle_items_equipped_total"] = int(self.battle_items_equipped_total)
        finalized_info["battle_items_used_total"] = int(self.battle_items_used_total)
        finalized_info["books_equipped_total"] = int(self.books_equipped_total)
        finalized_info["unit_swaps_total"] = int(self.grid_unit_swaps_total)
        finalized_info["book_equip_used_this_turn"] = bool(self.book_equip_used_this_turn)
        finalized_info["combat_potion_battle_bonus_pending"] = bool(
            self.combat_potion_battle_bonus_pending
        )
        finalized_info["scripted_capital_bot_enemies_defeated"] = int(
            getattr(self, "scripted_capital_bot_enemies_defeated", 0) or 0
        )

        include_detailed_info = self._include_detailed_step_info()
        if not include_detailed_info:
            return (
                self._build_obs(grid_obs=grid_obs),
                float(finalized_reward),
                bool(terminated),
                bool(truncated),
                finalized_info,
            )

        finalized_info["heroitems"] = list(self.heroitems)
        finalized_info["scenario_potion_item_names"] = list(self.scenario_potion_item_names)
        finalized_info["scenario_merchant_potion_item_names"] = list(
            self.scenario_merchant_potion_item_names
        )
        finalized_info["scenario_battle_potion_equip_names"] = list(
            self.scenario_battle_potion_equip_names
        )
        finalized_info["equipped_hero_items"] = list(self.equipped_hero_items)
        finalized_info["equipped_hero_item_uses_left"] = list(
            self.equipped_hero_item_uses_left
        )
        finalized_info.update(self._artifact_state_info())
        finalized_info.update(self._banner_state_info())
        finalized_info.update(self._book_state_info())
        finalized_info.update(self._boot_state_info())
        finalized_info["extra_heal_bottles"] = int(self.extra_heal_bottles or 0)
        finalized_info["extra_healing_bottles"] = int(self.extra_healing_bottles or 0)
        finalized_info["extra_revive_bottles"] = int(self.extra_revive_bottles or 0)
        finalized_info["invulnerability_potions_left"] = self._count_hero_item(
            self.INVULNERABILITY_POTION_ITEM_NAME
        )
        finalized_info["strength_potions_left"] = self._count_hero_item(
            self.STRENGTH_POTION_ITEM_NAME
        )
        finalized_info["energy_elixirs_left"] = self._count_hero_item(
            self.ENERGY_ELIXIR_ITEM_NAME
        )
        finalized_info["haste_elixirs_left"] = self._count_hero_item(
            self.HASTE_ELIXIR_ITEM_NAME
        )
        finalized_info["titan_elixirs_left"] = self._count_hero_item(
            self.TITAN_ELIXIR_ITEM_NAME
        )
        finalized_info["supreme_elixirs_left"] = self._count_hero_item(
            self.SUPREME_ELIXIR_ITEM_NAME
        )
        finalized_info["fire_wards_left"] = self._count_hero_item(self.FIRE_WARD_ITEM_NAME)
        finalized_info["earth_wards_left"] = self._count_hero_item(self.EARTH_WARD_ITEM_NAME)
        finalized_info["water_wards_left"] = self._count_hero_item(self.WATER_WARD_ITEM_NAME)
        finalized_info["air_wards_left"] = self._count_hero_item(self.AIR_WARD_ITEM_NAME)
        finalized_info["active_invulnerability_positions"] = sorted(
            self.active_invulnerability_potion_positions
        )
        finalized_info["active_strength_positions"] = sorted(
            self.active_strength_potion_positions
        )
        finalized_info["active_energy_positions"] = sorted(self.active_energy_elixir_positions)
        finalized_info["active_haste_positions"] = sorted(self.active_haste_elixir_positions)
        finalized_info["active_fire_ward_positions"] = sorted(self.active_fire_ward_positions)
        finalized_info["active_earth_ward_positions"] = sorted(self.active_earth_ward_positions)
        finalized_info["active_water_ward_positions"] = sorted(self.active_water_ward_positions)
        finalized_info["active_air_ward_positions"] = sorted(self.active_air_ward_positions)
        finalized_info["active_potion_effect_positions"] = {
            item_name: sorted(positions)
            for item_name, positions in self.active_potion_effect_positions.items()
            if positions
        }
        finalized_info.setdefault("collected_chests", [])
        finalized_info["chests_remaining"] = len(self.chests)
        finalized_info["legions_territory_tiles"] = tuple(self.legions_territory_tiles)
        finalized_info["legions_territory_count"] = len(self.legions_territory_tiles)
        finalized_info["legions_captured_gold_mine_tiles"] = tuple(
            self.legions_captured_gold_mine_tiles
        )
        finalized_info["legions_captured_gold_mine_count"] = int(
            self.legions_captured_gold_mine_count
        )
        finalized_info["legions_gold_mine_income_per_turn"] = (
            int(self.legions_captured_gold_mine_count)
            * float(self.LEGIONS_GOLD_MINE_GOLD_PER_TURN)
        )
        finalized_info["empire_territory_tiles"] = tuple(self.empire_territory_tiles)
        finalized_info["empire_territory_count"] = len(self.empire_territory_tiles)
        finalized_info.update(self._legions_settlement_territory_info())
        finalized_info.update(self._mana_state_info())
        finalized_info.update(self._spell_state_info())
        finalized_info.update(self._ruin_progress_info())
        finalized_info.update(self._scripted_capital_bot_info())
        finalized_info.update(self._merchant_context_info(position))
        finalized_info.update(self._spell_shop_context_info(position))
        finalized_info.update(self._mercenary_context_info(position))
        finalized_info.update(self._trainer_context_info(position))

        return (
            self._build_obs(grid_obs=grid_obs),
            float(finalized_reward),
            bool(terminated),
            bool(truncated),
            finalized_info,
        )
    def _log(self, message: str):
        """Записать строку в буфер логов кампании и при необходимости вывести её.

        Буфер используется визуализацией и диагностикой, а log_enabled включает
        немедленную печать в stdout без изменения логики среды.
        """
        self._campaign_logs.append(message)
        if self.log_enabled:
            print(f"[CAMPAIGN] {message}")
    def pop_campaign_logs(self) -> List[str]:
        """Вернуть накопленные campaign-логи и очистить внутренний буфер."""
        logs = self._campaign_logs[:]
        self._campaign_logs.clear()
        return logs
    def pop_battle_logs(self) -> List[str]:
        """Забрать pretty-events из текущего BattleEnv, если бой активен."""
        if self.battle_env is not None:
            return self.battle_env.pop_pretty_events()
        return []
    def render(self, mode: str = "ansi") -> Optional[str]:
        """Build a compact text summary of the current campaign state."""
        if mode != "ansi":
            return None

        self._refresh_campaign_equipment_effects(log=False)
        self._sync_equipped_book_items()
        self._sync_equipped_boot_items()
        lines = []
        lines.append(f"=== CAMPAIGN MODE: {'GRID' if self.mode == self.MODE_GRID else 'BATTLE'} ===")

        if self.mode == self.MODE_GRID:
            lines.append("")
            lines.append(f"Agent: {self.grid_env.agent_pos}")
            lines.append(f"Enemies alive: {[k for k, v in self.grid_env.enemies_alive.items() if v]}")
            lines.append(f"Visited: {len(self.grid_env.visited_cells)}/{self.grid_size**2}")
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")
        else:
            lines.append(f"В бою с врагом {self.current_enemy_id}")
            lines.append(f"Описание: {ENEMY_DESCRIPTIONS.get(self.current_enemy_id, '?')}")
            lines.append(f"Ход: {self.turns} | Золото: {self.gold:g} | Шаги: {self.moves}")
        mana_totals = self._current_mana_totals()
        mana_income = self._compute_mana_income_per_turn()
        mana_text = ", ".join(
            f"{self.MANA_LABEL_BY_KIND[str(kind)]}: {mana_totals[str(kind)]:g} (+{mana_income[str(kind)]:g}/ход)"
            for kind in self.MANA_KIND_ORDER
        )
        lines.append(f"Мана: {mana_text}")
        lines.append(
            "Башня магии: "
            + ("построена" if self._has_magic_tower_built() else "не построена")
        )
        lines.append(
            f"Изучение заклинаний в этом ходу: "
            + ("заблокировано" if self.spell_learning_locked else "доступно")
        )

        if self.chests:
            chest_text = ", ".join(
                f"{pos}:{list(item_names)}" for pos, item_names in sorted(self.chests.items())
            )
            lines.append(f"Сундуки: {chest_text}")
        else:
            lines.append("Сундуки: нет")
        if self.heroitems:
            lines.append(
                "Предметы героя: "
                + ", ".join(self._hero_item_display_name(entry) for entry in self.heroitems)
            )
        else:
            lines.append("Предметы героя: нет")
        lines.append(
            "Маг для свитков: "
            + ("нанят" if self.scroll_magic_unlocked else "не нанят")
        )
        scroll_state = self._scroll_state_info()
        scroll_inventory = list(scroll_state.get("scroll_inventory", []))
        if scroll_inventory:
            lines.append(
                "Свитки в инвентаре: "
                + ", ".join(
                    f"{entry['spell_name']} x{int(entry['copies'])}"
                    + ("" if entry.get("supported", False) else " (unsupported)")
                    for entry in scroll_inventory
                )
            )
        else:
            lines.append("Свитки в инвентаре: нет")
        scroll_slot_entries = self.scroll_cast_slot_entries()
        if scroll_slot_entries:
            lines.append("Слоты scroll-action:")
            lines.extend(
                f"- slot {idx}: {entry['spell_name']} x{int(entry['copies'])}"
                for idx, entry in enumerate(scroll_slot_entries)
            )
        equipped_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_hero_items or [])
        ]
        lines.append(f"Экипированные предметы: [{', '.join(equipped_labels)}]")
        artifact_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_artifact_items or [])
        ]
        lines.append(f"Экипированные артефакты: [{', '.join(artifact_labels)}]")
        banner_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_banner_items or [])
        ]
        lines.append(f"Экипированное знамя: [{', '.join(banner_labels)}]")
        book_labels = [
            str(item_name or "empty") for item_name in list(self.equipped_book_items or [])
        ]
        lines.append(f"Equipped book: [{', '.join(book_labels)}]")
        boot_labels = [
            str(item_name or "пусто") for item_name in list(self.equipped_boot_items or [])
        ]
        lines.append(f"Экипированные сапоги: [{', '.join(boot_labels)}]")
        lines.append(
            f"Боевые предметы: экипировано {int(self.battle_items_equipped_total)}, "
            f"использовано {int(self.battle_items_used_total)}"
        )
        lines.append(f"Доп. бутылей лечения: {max(0, int(self.extra_heal_bottles or 0))}")
        lines.append(f"Доп. банок исцеления: {max(0, int(self.extra_healing_bottles or 0))}")
        lines.append(f"Доп. зелий воскрешения: {max(0, int(self.extra_revive_bottles or 0))}")
        lines.append(
            "Зелий неуязвимости: "
            f"{self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME)}"
        )
        lines.append(
            "Зелий силы: "
            f"{self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME)}"
        )
        if self.active_invulnerability_potion_positions:
            lines.append(
                "Активное зелье неуязвимости на позициях: "
                f"{sorted(self.active_invulnerability_potion_positions)}"
            )
        if self.active_strength_potion_positions:
            lines.append(
                "Активное зелье силы на позициях: "
                f"{sorted(self.active_strength_potion_positions)}"
            )
        learned_spells = self.get_learned_spell_descriptions()
        if learned_spells:
            lines.append(
                f"Изученные заклинания ({len(learned_spells)}/{len(self.spell_keys)}):"
            )
            lines.extend(f"- {description}" for description in learned_spells)
        else:
            lines.append(f"Изученные заклинания: 0/{len(self.spell_keys)}")

        result = "\n".join(lines)
        print(result)
        return result
    def get_state_summary(self) -> Dict:
        """Вернуть машинно-читаемый snapshot кампании для UI и диагностики.

        В отличие от step info, summary всегда собирает широкий срез состояния:
        режим, позицию, врагов, ресурсы, инвентарь, экипировку, книги/ботинки,
        магию, магазины, наёмников, ману и оставшиеся сундуки. Перед сбором
        синхронизируются эффекты экипировки, чтобы snapshot отражал текущие
        бонусы, а не устаревший кеш.
        """
        self._refresh_campaign_equipment_effects(log=False)
        self._sync_equipped_book_items()
        self._sync_equipped_boot_items()
        scroll_state = self._scroll_state_info()
        book_state = self._book_state_info()
        boot_state = self._boot_state_info()
        return {
            "mode": "grid" if self.mode == self.MODE_GRID else "battle",
            "agent_pos": self.grid_env.agent_pos,
            "enemies_alive": dict(self.grid_env.enemies_alive),
            "enemies_defeated": [
                eid for eid, alive in self.grid_env.enemies_alive.items() if not alive
            ],
            "visited_cells": len(self.grid_env.visited_cells),
            "grid_steps": self.grid_env.step_count,
            "current_enemy": self.current_enemy_id,
            "blue_hp_saved": self.blue_team_state is not None,
            "turns": self.turns,
            "gold": self.gold,
            "moves": self.moves,
            "typeoflord": int(self.typeoflord),
            "heroitems": list(self.heroitems),
            "scenario_potion_item_names": list(self.scenario_potion_item_names),
            "scenario_merchant_potion_item_names": list(self.scenario_merchant_potion_item_names),
            "scenario_battle_potion_equip_names": list(self.scenario_battle_potion_equip_names),
            "equipped_hero_items": list(self.equipped_hero_items),
            "equipped_hero_item_uses_left": list(self.equipped_hero_item_uses_left),
            "equipped_artifact_items": list(self.equipped_artifact_items),
            "equipped_banner_items": list(self.equipped_banner_items),
            "equipped_book_items": list(book_state["equipped_book_items"]),
            "equipped_boot_items": list(boot_state["equipped_boot_items"]),
            "active_boot_move_bonus": int(boot_state["active_boot_move_bonus"]),
            "artifact_auto_equip_next_slot": int(self.artifact_auto_equip_next_slot),
            "book_equip_used_this_turn": bool(book_state["book_equip_used_this_turn"]),
            "battle_items_equipped_total": int(self.battle_items_equipped_total),
            "battle_items_used_total": int(self.battle_items_used_total),
            "books_equipped_total": int(book_state["books_equipped_total"]),
            "unit_swaps_total": int(self.grid_unit_swaps_total),
            "learned_spells": list(self.get_learned_spell_descriptions()),
            "learned_spells_count": len(self.get_learned_spell_descriptions()),
            "spells_total": len(self.spell_keys),
            "spell_learning_locked": bool(self.spell_learning_locked),
            "magic_tower_built": bool(self._has_magic_tower_built()),
            "scroll_magic_unlocked": bool(self.scroll_magic_unlocked),
            "scroll_inventory": list(scroll_state.get("scroll_inventory", [])),
            "scroll_cast_slots": list(scroll_state.get("scroll_cast_slots", [])),
            "supported_scroll_types_count": int(scroll_state.get("supported_scroll_types_count", 0) or 0),
            "total_scroll_count": int(scroll_state.get("total_scroll_count", 0) or 0),
            "extra_heal_bottles": int(self.extra_heal_bottles or 0),
            "extra_healing_bottles": int(self.extra_healing_bottles or 0),
            "extra_revive_bottles": int(self.extra_revive_bottles or 0),
            "invulnerability_potions_left": self._count_hero_item(self.INVULNERABILITY_POTION_ITEM_NAME),
            "strength_potions_left": self._count_hero_item(self.STRENGTH_POTION_ITEM_NAME),
            "active_invulnerability_positions": sorted(self.active_invulnerability_potion_positions),
            "active_strength_positions": sorted(self.active_strength_potion_positions),
            "active_potion_effect_positions": {
                item_name: sorted(positions)
                for item_name, positions in self.active_potion_effect_positions.items()
                if positions
            },
            "merchant_stocks": {
                site_name: self._merchant_stock_snapshot(site_name)
                for site_name in self.merchant_stocks.keys()
            },
            "spell_shop_stocks": {
                site_name: self._spell_shop_stock_snapshot(site_name)
                for site_name in self.spell_shop_stocks.keys()
            },
            "spell_shop_sites_in_range": list(self._spell_shop_sites_at_position()),
            "mercenary_stocks": {
                site_name: self._mercenary_stock_snapshot(site_name)
                for site_name in self.mercenary_site_rosters.keys()
            },
            "trainer_sites_in_range": list(self._trainer_sites_at_position()),
            **self._mana_state_info(),
            "chests_remaining": [
                {"pos": pos, "items": list(item_names)}
                for pos, item_names in sorted(self.chests.items())
            ],
        }
