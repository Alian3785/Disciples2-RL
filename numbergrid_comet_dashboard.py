"""Compact Comet charts; raw campaign telemetry stays in charts.jsonl.

The chart catalog and Russian titles here also define the W&B view
(numbergrid_wandb_dashboard.py). Comet selects panels for the map objective.
"""
from copy import deepcopy
from uuid import uuid4

# Means include completed episodes only. Battle victories, hiring and potion use show the last
# 100 episodes; the other mechanics retain their run-long means.
RECENT_MECHANICS = frozenset(('battle_victories', 'hired_units', 'mercenaries_hired', 'potions_used'))
CAMPAIGN_MECHANICS = (
    'battle_victories', 'battle_defeats', 'battle_retreats', 'unit_upgrades',
    'castle_heal_uses', 'castle_healed_hp', 'chests_collected', 'buildings_built',
    'ruins_cleared', 'potions_used', 'merchant_purchases', 'merchant_sale_gold', 'spells_learned',
    'spell_casts', 'summoned_units', 'hired_units', 'mercenaries_hired', 'battle_items_equipped',
    'battle_items_used', 'unit_swaps', 'enemy_squads_defeated', 'cities_captured', 'trainer_uses',
    'capital_defense_effects', 'capital_health_effects',
    'capital_guardian_rounds', 'capital_guardian_battles',
    'capital_entry_strength', 'capital_guardian_bonuses',
)
CAMPAIGN_CHARTS = (
    # First row (three per row): run-long win and defeat rates, then the
    # latest interval's win rate in place of the run-long timeout rate.
    'campaign/victory_rate', 'campaign/defeat_rate', 'campaign/window/victory_rate',
    'campaign/mean_seconds_per_million_steps',
    'campaign/battle_win_rate',
    *(f"campaign/{'window' if name in RECENT_MECHANICS else 'completed'}/{name}_mean"
      for name in CAMPAIGN_MECHANICS),
    'campaign/timeout_rate', 'campaign/window/battle_win_rate',
    'campaign/window/unit_upgrades_mean',
    'campaign/window/experience_gained_mean', 'campaign/window/waves_completed_mean',
    'campaign/window/dragon_reached_episodes_rate', 'campaign/victories_total',
    'campaign/window/capital_defense_acquired_mean',
    'campaign/window/capital_guardian_reached_episodes_rate',
    'campaign/window/capital_defense_effects_when_reached_mean',
    'campaign/window/capital_health_effects_when_reached_mean',
    'campaign/window/capital_guardian_rounds_when_reached_mean',
    'campaign/window/capital_entry_strength_when_reached_mean',
    'campaign/window/capital_bot_heroes_defeated_mean', 'campaign/window/capital_incubi_mean',
)
CHART_TITLES = {
    'campaign/mean_seconds_per_million_steps': 'Секунд на 1 млн шагов — в среднем с начала запуска',
    'campaign/victory_rate': 'Доля побед на карте — с начала обучения',
    'campaign/defeat_rate': 'Доля поражений на карте — с начала обучения',
    'campaign/timeout_rate': 'Доля завершений по лимиту шагов — с начала обучения',
    'campaign/battle_win_rate': 'Доля побед в боях — с начала обучения',
    'campaign/completed/battle_defeats_mean': 'Поражений в боях — в среднем на эпизод',
    'campaign/completed/battle_retreats_mean': 'Отступлений из боя — в среднем на эпизод',
    'campaign/completed/unit_upgrades_mean': 'Улучшений юнитов — в среднем на эпизод',
    'campaign/completed/castle_heal_uses_mean': 'Применений лечения в столице — в среднем на эпизод',
    'campaign/completed/castle_healed_hp_mean': 'Восстановлено здоровья в столице — в среднем на эпизод',
    'campaign/completed/chests_collected_mean': 'Собрано сундуков — в среднем на эпизод',
    'campaign/completed/buildings_built_mean': 'Построено зданий — в среднем на эпизод',
    'campaign/completed/ruins_cleared_mean': 'Разграблено руин — в среднем на эпизод',
    'campaign/window/potions_used_mean': 'Использовано зелий на эпизод — последние 100 эпизодов',
    'campaign/completed/merchant_purchases_mean': 'Покупок у торговца — в среднем на эпизод',
    'campaign/completed/merchant_sale_gold_mean': 'Золото от продажи вещей — в среднем на эпизод',
    'campaign/completed/spells_learned_mean': 'Изучено заклинаний — в среднем на эпизод',
    'campaign/completed/spell_casts_mean': 'Применено заклинаний — в среднем на эпизод',
    'campaign/completed/summoned_units_mean': 'Призвано существ на карте — в среднем на эпизод',
    'campaign/window/hired_units_mean': 'Нанято бойцов на эпизод — последние 100 эпизодов',
    'campaign/window/mercenaries_hired_mean': 'Нанято наёмников на эпизод — последние 100 эпизодов',
    'campaign/completed/trainer_uses_mean': 'Обучений у тренера — в среднем на эпизод',
    'campaign/completed/battle_items_equipped_mean': 'Экипировано боевых предметов — в среднем на эпизод',
    'campaign/completed/battle_items_used_mean': 'Использовано боевых предметов — в среднем на эпизод',
    'campaign/completed/unit_swaps_mean': 'Перестановок бойцов — в среднем на эпизод',
    'campaign/completed/enemy_squads_defeated_mean': 'Уничтожено вражеских отрядов — в среднем на эпизод',
    'campaign/completed/cities_captured_mean': 'Захвачено городов — в среднем на эпизод',
    'campaign/window/victory_rate': 'Доля побед на карте — за последний интервал',
    'campaign/window/battle_win_rate': 'Доля побед в боях — за последний интервал',
    'campaign/window/battle_victories_mean': 'Побед в боях на эпизод — последние 100 эпизодов',
    'campaign/window/unit_upgrades_mean': 'Улучшений юнитов на эпизод — последние 100 эпизодов',
    'campaign/window/waves_completed_mean': 'Пройдено волн на эпизод — последние 100 эпизодов',
    'campaign/window/dragon_reached_episodes_rate': 'Доля эпизодов с боем против дракона — последние 100 эпизодов',
    'campaign/victories_total': 'Побед на карте — всего с начала обучения',
    'campaign/completed/capital_defense_effects_mean': 'Защитных эффектов у Мизраэля, включая HP — на эпизод',
    'campaign/completed/capital_health_effects_mean': 'Эффектов постоянного HP у Мизраэля — на эпизод',
    'campaign/completed/capital_guardian_rounds_mean': 'Новых раундов с Мизраэлем — на эпизод',
    'campaign/completed/capital_guardian_battles_mean': 'Вступлений в бой с Мизраэлем — на эпизод',
    'campaign/window/capital_defense_acquired_mean': 'Получено зелий защиты, HP, точности и инициативы — последние 100 эпизодов',
    'campaign/window/capital_guardian_reached_episodes_rate': 'Дошли до Мизраэля — последние 100 эпизодов',
    'campaign/window/capital_defense_effects_when_reached_mean': 'Защитных эффектов — в эпизодах с Мизраэлем (последние 100)',
    'campaign/window/capital_health_effects_when_reached_mean': 'Эффектов постоянного HP — в эпизодах с Мизраэлем (последние 100)',
    'campaign/window/capital_guardian_rounds_when_reached_mean': 'Новых раундов — в эпизодах с Мизраэлем (последние 100)',
    'campaign/completed/capital_entry_strength_mean': 'Сила армии при первом входе к Мизраэлю — на эпизод',
    'campaign/completed/capital_guardian_bonuses_mean': 'Бой с Мизраэлем после 70+ побед (+10) — доля эпизодов',
    'campaign/window/capital_bot_heroes_defeated_mean': 'Побед над героями бота Империи на эпизод — последние 100 эпизодов',
    'campaign/window/experience_gained_mean': 'Опыт, полученный бойцами агента, на эпизод — последние 100 эпизодов',
    'campaign/window/capital_incubi_mean': 'Инкубов в отряде — максимум за эпизод, последние 100 эпизодов',
    'campaign/window/capital_entry_strength_when_reached_mean': 'Сила армии при первом входе — среди дошедших (последние 100)',
    'rollout/ep_rew_mean': 'Средняя награда — последние 100 эпизодов',
    'rollout/ep_len_mean': 'Средняя длина — последние 100 эпизодов',
}
ROLLOUT_CHARTS = ('rollout/ep_rew_mean', 'rollout/ep_len_mean')
# Collapsed sections below rollout and campaign; the W&B view reuses them.
TECHNICAL_SECTIONS = (
    ('performance', tuple(f'performance/{name}' for name in (
        'steps_per_second', 'mean_steps_per_second', 'training_seconds',
        'loop_steps_per_second'))),
    ('train', tuple(f'train/{name}' for name in (
        'policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl',
        'clip_fraction', 'explained_variance'))),
    *((group, tuple(f'{group}/{name}' for name in (
        'return_mean', 'win_rate', 'length_mean', 'wins', 'episodes')))
      for group in ('eval_argmax', 'eval_sample')),
)
VIEW_NAME = 'Disciples II'
CAPITAL_VIEW_NAME = 'Disciples II — Мизраэль'
CAPITAL_CHARTS = frozenset(name for name in CAMPAIGN_CHARTS
                         if name.rsplit('/', 1)[-1].startswith('capital_'))


def comet_campaign_charts(game_map=None):
    """The presence of a guardian on the map is not enough: it must be the goal."""
    if (game_map or {}).get('campaign_objective') == 'capital_guardian':
        return CAMPAIGN_CHARTS
    return tuple(name for name in CAMPAIGN_CHARTS if name not in CAPITAL_CHARTS)


def comet_view_name(game_map=None):
    return (CAPITAL_VIEW_NAME if (game_map or {}).get('campaign_objective') == 'capital_guardian'
            else VIEW_NAME)


def compact_comet_metrics(data, game_map=None):
    charts = comet_campaign_charts(game_map)
    return {key: value for key, value in data.items()
            if not key.startswith('campaign/') or key in charts}


def line_panel(metric):
    return {
        'chartId': uuid4().hex, 'chartName': CHART_TITLES.get(metric, metric), 'chartType': 'line',
        'metricName': metric, 'autoGenerated': False, 'selectedXAxis': 'step',
        'selectedYAxis': metric, 'metricNames': [metric],
        'selectedOutliers': 'show', 'selectedLegendKeys': [
            {'value': 'Name', 'label': 'Name', 'source': 'log_other'}],
        'transformX': None, 'transformY': None, 'smoothingY': 0, 'sampleSize': 500,
        'fillType': 'LINE', 'aggregationX': 'last', 'hoverMode': 'closest',
        'grouping': {'enabled': False, 'aggregation': 'mean', 'range': 'none',
                     'groupByParameter': None},
        'customRange': {'x': {'min': '', 'max': ''}, 'y': {'min': '', 'max': ''}},
        'legendMode': 'auto', 'locked': False,
        'customXAxisTitle': '', 'showXAxisTitle': False, 'showXAxisTickLabels': True,
        'customYAxisTitle': '', 'showYAxisTitle': False, 'showYAxisTickLabels': True,
    }


def chart_section(title, metrics, expanded=False):
    panels = [line_panel(metric) for metric in metrics]
    return {'id': uuid4().hex, 'title': title, 'autogenerated': False,
            'expanded': expanded, 'panels': panels, 'layout': [
                {'i': panel['chartId'], 'minW': 2, 'x': (index % 3) * 2,
                 'y': index // 3, 'w': 2, 'h': 1, 'static': False,
                 'moved': False, 'maxH': 3}
                for index, panel in enumerate(panels)]}


def compact_comet_view(base=None, game_map=None):
    """Comet SDK 3.58.8's v2 / panel version 5 state, preserving other sections."""
    state = deepcopy(base) if base else {}
    state.update(template_name=comet_view_name(game_map), x_axis='step', smoothing=0)
    # Use a named saved view: the REST API cannot update the UI's personal draft.
    state.pop('template_id', None)
    state['autoGeneratedDraft'] = False
    for key, value in {'charts': [], 'layout': [], 'sampleSize': 500,
                       'entire_row_charts': False,
                       'outliers': {'percentile': 0, 'isVisible': True}}.items():
        state.setdefault(key, value)
    panels = state.setdefault('panels', {})
    previous = panels.get('sections', [])
    other_sections = [s for s in previous if s['title'] not in ('rollout', 'campaign')]
    if not previous:
        other_sections = [chart_section(title, metrics) for title, metrics in TECHNICAL_SECTIONS]
    rollout = chart_section('rollout', ROLLOUT_CHARTS, expanded=True)
    rollout['layout'][0].update(x=0, y=0, w=4)
    rollout['layout'][1].update(x=4, y=0, w=2)
    panels.update(version=5, isAutoRefreshEnabled=True, compareXAxis='step',
                  autoGenerate=False, sections=[
                      rollout, chart_section('campaign', comet_campaign_charts(game_map), expanded=True),
                      *other_sections])
    panels.setdefault('layout', [])
    panels.setdefault('panels', [])
    return state


def configure_comet_dashboard(api_experiment, game_map=None):
    """Save the compact view; callers link to its explicit viewId.

    The public create_view method strips template_id even for updates, so use
    the SDK upsert helper to reuse our named view without creating duplicates.
    Personal UI drafts are left intact: Comet's REST upsert rejects their IDs.
    Named templates are shared by runs, so guardian objectives use a separate
    template instead of toggling panels in the general view on every run.
    """
    from comet_ml.api_objects.view import View

    views = [v for v in api_experiment.get_views() if v.view_source == 'chart_template']
    name = comet_view_name(game_map)
    charts = comet_campaign_charts(game_map)
    view = next((v for v in views if v.name == name and not v.unsaved_view), None)
    if view is None:
        base = next((v for v in views if v.name == VIEW_NAME and not v.unsaved_view), None)
        if base is None:
            base = next((v for v in views if v.current or v.unsaved_view), None)
        saved = api_experiment.create_view(View(
            name=name, v2=compact_comet_view(base.v2 if base else None, game_map)))
    else:
        view.v2 = compact_comet_view(view.v2, game_map)
        view.experiment_key = api_experiment.id
        payload = api_experiment._api._client.upsert_chart_template_view(
            api_experiment.project_id, view)
        saved = View.from_payload_dict(payload) if payload else None
    if not saved:
        raise RuntimeError('Comet did not confirm the dashboard update')
    if not saved.template_id:
        raise RuntimeError('Comet did not return the saved view ID')
    return {'status': 'saved', 'name': name, 'view_id': saved.template_id,
            'first_section': 'rollout',
            'first_metric': ROLLOUT_CHARTS[0], 'campaign_charts': len(charts),
            'campaign_window_charts': sum(metric.startswith('campaign/window/') for metric in charts)}
