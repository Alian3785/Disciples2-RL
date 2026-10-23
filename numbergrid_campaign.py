"""Reference campaign charts from GPU-reduced counters, once per learner block."""
from collections import deque

from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, EPISODE_WINDOW

PARTICIPATION_NAMES = {
    'castle_heal_uses': 'castle_heal', 'chests_collected': 'chest_collection',
    'buildings_built': 'building', 'ruins_cleared': 'ruin_clear',
    'sold_items': 'sale', 'spells_learned': 'spell_learning', 'spell_casts': 'spell_cast',
    'summoned_units': 'summon', 'hired_units': 'hire', 'mercenaries_hired': 'mercenary_hire',
    'battle_items_equipped': 'battle_item_equip', 'battle_items_used': 'battle_item_use',
    'unit_swaps': 'unit_swap', 'trainer_uses': 'trainer_use', 'merchant_purchases': 'merchant_purchase',
    'potions_used': 'potion_use', 'dragon_battles': 'dragon_reached',
    'capital_guardian_battles': 'capital_guardian_reached',
}


class CampaignMetrics:
    def __init__(self, waves_enabled=False):
        self.waves_enabled = waves_enabled
        self.total = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
        self.window = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
        self.participation = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
        self.completed = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
        self.recent = deque(maxlen=EPISODE_WINDOW)
        self.best_recent_victory_rate = 0.
        self.active = False

    def add(self, summary):
        """Called after device_get, never from an individual environment step."""
        self.active = True
        for index, name in enumerate(CAMPAIGN_FIELDS):
            event = float(summary['events'][index])
            self.total[name] += event
            self.window[name] += event
            self.participation[name] += float(summary['participation'][index])
            self.completed[name] += float(summary['completed'][index])
        count = min(int(summary['count']), EPISODE_WINDOW)
        if count:
            self.recent.extend([list(map(float, record)) for record in summary['recent'][-count:]])

    def metrics(self, reset_window=False):
        if not self.active:
            return {}
        total, window = self.total, self.window
        episodes = sum(total[name] for name in ('victories', 'defeats', 'timeouts'))
        window_episodes = sum(window[name] for name in ('victories', 'defeats', 'timeouts'))
        battles = sum(total[name] for name in ('battle_victories', 'battle_defeats', 'battle_retreats'))
        window_battles = sum(window[name] for name in ('battle_victories', 'battle_defeats', 'battle_retreats'))
        fields = [(index, name) for index, name in enumerate(CAMPAIGN_FIELDS)
                  if name != 'waves_completed' or self.waves_enabled]
        data = {f'campaign/{name}_total': total[name] for _, name in fields}
        data.update({'campaign/episodes_total': episodes,
                     'campaign/window/episodes': window_episodes,
                     'campaign/window/enemy_squads_defeated_sample_count': len(self.recent)})
        for scope, counts, denominator in (('', total, episodes), ('window/', window, window_episodes)):
            if denominator:
                for singular, plural in (('victory', 'victories'), ('defeat', 'defeats'), ('timeout', 'timeouts')):
                    data[f'campaign/{scope}{singular}_rate'] = counts[plural] / denominator
        for scope, counts, denominator in (('', total, battles), ('window/', window, window_battles)):
            if denominator:
                data[f'campaign/{scope}battle_win_rate'] = counts['battle_victories'] / denominator
        if window_episodes:
            self.best_recent_victory_rate = max(self.best_recent_victory_rate, window['victories'] / window_episodes)
        data['campaign/best_recent_victory_rate'] = self.best_recent_victory_rate
        if episodes:
            # The reference's total/episodes includes events in still-running
            # episodes. Also expose an explicitly completed-only mean.
            for _, name in fields[3:]:
                data[f'campaign/{name}_per_episode_mean'] = total[name] / episodes
                data[f'campaign/completed/{name}_mean'] = self.completed[name] / episodes
            for name, alias in PARTICIPATION_NAMES.items():
                data[f'campaign/{alias}_episodes_rate'] = self.participation[name] / episodes
        if self.recent:
            for index, name in fields[3:]:
                data[f'campaign/window/{name}_mean'] = sum(record[index] for record in self.recent) / len(self.recent)
            data['campaign/window/enemy_squads_defeated_mean_100'] = data['campaign/window/enemy_squads_defeated_mean']
            data['campaign/window/episode_return_mean'] = sum(record[-2] for record in self.recent) / len(self.recent)
            data['campaign/window/episode_length_mean'] = sum(record[-1] for record in self.recent) / len(self.recent)
            dragon = CAMPAIGN_FIELDS.index('dragon_battles')
            data['campaign/window/dragon_reached_episodes_rate'] = sum(
                record[dragon] > 0 for record in self.recent) / len(self.recent)
            guardian = CAMPAIGN_FIELDS.index('capital_guardian_battles')
            reached = [record for record in self.recent if record[guardian] > 0]
            data['campaign/window/capital_guardian_reached_episodes_rate'] = len(reached) / len(self.recent)
            if reached:
                for name in ('capital_defense_effects', 'capital_health_effects', 'capital_guardian_rounds'):
                    index = CAMPAIGN_FIELDS.index(name)
                    data[f'campaign/window/{name}_when_reached_mean'] = sum(r[index] for r in reached) / len(reached)
            first = CAMPAIGN_FIELDS.index('capital_first_contacts')
            contacts = [record for record in self.recent if record[first] > 0]
            if contacts:
                strength = CAMPAIGN_FIELDS.index('capital_entry_strength')
                data['campaign/window/capital_entry_strength_when_reached_mean'] = (
                    sum(r[strength] for r in contacts) / len(contacts))
        if reset_window:
            self.window = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
        return data
