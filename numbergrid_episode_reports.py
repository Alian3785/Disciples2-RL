"""Readable episode reports from sparse, already transferred GPU batches."""
from collections import Counter
import gzip
import io
import json
from pathlib import Path

from stoix.envs.number_grid_episode_schema import HEADER, SCHEMA_VERSION, WIDTH

OUTCOMES = ('ongoing', 'victory', 'defeat', 'timeout')
METHODS = ('none', 'hero', 'summon', 'spell')
BATTLES = ('none', 'victory', 'defeat', 'retreat')


def read_episode_batch(packets, count):
    """Transfer a populated prefix once per block, using reusable slice shapes."""
    import jax  # The separate CPU writer never imports or initializes JAX.

    if count == 0:
        return {name: [] for name in packets}
    if not 0 < count <= packets['data'].shape[0]:
        raise ValueError('Invalid GPU episode event count')
    capacity = min(1 << (count - 1).bit_length(), packets['data'].shape[0])
    host = jax.device_get(jax.tree.map(lambda x: x[:capacity], packets))
    return {name: values[:count] for name, values in host.items()}


class EpisodeReports:
    def __init__(self, output, game_map, combat, phase='training'):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.game_map, self.combat, self.phase = game_map, combat, phase
        self.catalogue = combat['catalogue']
        self.active, self.sequences = {}, Counter()
        self.completed = self.event_count = self.xp_total = 0
        self.outcomes, self.formations, self.units = Counter(), Counter(), Counter()
        self.defeated, self.killers = Counter(), Counter()
        self.path = self.output / 'episode_reports.jsonl.gz'
        if self.path.exists():
            raise FileExistsError(f'Episode reports already exist: {self.path}')
        self.path.touch()
        self._buffer = io.BytesIO()
        self._gzip = None
        self._pending_records = 0
        self.uncompressed_bytes = 0
        self._json('episode_reports.metadata.json', dict(schema_version=SCHEMA_VERSION,
            phase=phase, map=game_map['name'],
            source='Shared JAX environment; sparse events before autoreset',
            experience='Awarded XP before level-up reset/cap; battle and trainer XP separated',
            fighter_identity='Episode-local identity follows swaps, including whole-column swaps',
            event_order='decision order; simultaneous promotions ordered by final slot',
            catalogue=self.catalogue, enemies=combat['enemies'],
            reference_enemy_ids=game_map.get('reference_enemy_ids'), complete_coverage=True,
            storage=dict(format='jsonl.gz', compression='gzip', compression_level=6,
                flush='On summary: every 1,000,000 training transitions and at completion')))

    def _json(self, name, data):
        temporary = self.output / (name + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.output / name)

    def _unit(self, party, slot, fighter_id, required=None):
        uid = int(party[0][slot])
        if not uid:
            return None
        profile = self.catalogue[uid]
        return dict(fighter_id=fighter_id, slot=slot+1, unit_id=uid,
            key=profile['key'], name=profile['name'], level=int(party[1][slot]),
            xp=int(party[2][slot]), xp_required=None if required is None else int(required[slot]),
            alive=bool(party[3][slot]))

    def _army(self, index, levels):
        if index < 0:
            return None
        refs = self.game_map.get('reference_enemy_ids')
        roster = [dict(slot=i+1, name=unit['name'], level=int(levels[i]))
                  for i, unit in enumerate(self.combat['enemies'][index]) if unit]
        return dict(index=index, number=index+1,
            reference_id=refs[index] if refs is not None else None, units=roster)

    def _new_fighter(self, episode, party, slot, step):
        fid = len(episode['fighters']) + 1
        unit = self._unit(party, slot, fid)
        episode['fighters'].append(dict(fighter_id=fid, joined_step=step,
            initial=unit, last=unit, battle_xp_received=0, trainer_xp_received=0,
            xp_received=0, in_final_party=True))
        return fid

    def _start(self, stream, old):
        self.sequences[stream] += 1
        episode = dict(schema_version=SCHEMA_VERSION, phase=self.phase,
            map=self.game_map['name'], stream=stream, episode=self.sequences[stream],
            fighters=[], initial_party=[], final_party=[], upgrades=[], experience_events=[],
            defeated_armies=[], battles=[], outcome='ongoing', defeat=None, _slots=[None]*6)
        for slot in range(6):
            if old[0][slot]:
                fid = self._new_fighter(episode, old, slot, 0)
                episode['_slots'][slot] = fid
                episode['initial_party'].append(self._unit(old, slot, fid))
        self.active[stream] = episode
        return episode

    def consume(self, batch, transition_base=0):
        """No device access: called once per learner block / diagnostic batch."""
        lines = []
        for values, reward, stream, offset in zip(batch['data'], batch['returns'],
                                                 batch['stream'], batch['offset']):
            row = [int(x) for x in values]
            if len(row) != WIDTH:
                raise ValueError('Episode event schema mismatch')
            event = dict(zip(HEADER, row[:len(HEADER)]))
            p = len(HEADER)
            old = [row[p+i*6:p+(i+1)*6] for i in range(4)]
            p += 24
            new = [row[p+i*6:p+(i+1)*6] for i in range(4)]
            p += 24
            required, gains, order, enemy_levels = (row[p+i*6:p+(i+1)*6] for i in range(4))
            stream = int(stream)
            episode = self.active.get(stream)
            if episode is None:
                if event['step'] != 1:
                    raise ValueError(f'Missing beginning of episode in stream {stream}')
                episode = self._start(stream, old)
            elif event['step'] <= episode['last_recorded_step']:
                raise ValueError(f'Out-of-order episode events in stream {stream}')
            episode['last_recorded_step'] = event['step']
            episode['last_recorded_day'] = event['day']
            episode['last_recorded_transition'] = transition_base + int(offset) + 1
            slots = [episode['_slots'][i] for i in order]
            old = [[part[i] for i in order] for part in old]
            army = self._army(event['enemy'], enemy_levels)
            method = METHODS[event['method']]
            episode['final_party'] = []
            for slot in range(6):
                fid = slots[slot]
                if not new[0][slot]:
                    if fid is not None:
                        episode['fighters'][fid-1]['last'] = self._unit(old, slot, fid)
                        episode['fighters'][fid-1]['in_final_party'] = False
                        episode['fighters'][fid-1]['left_step'] = event['step']
                    slots[slot] = None
                    continue
                if fid is None:
                    fid = self._new_fighter(episode, new, slot, event['step'])
                    slots[slot] = fid
                fighter = episode['fighters'][fid-1]
                unit = self._unit(new, slot, fid, required)
                fighter['last'] = unit
                source = 'battle' if event['battle_result'] and method == 'hero' else 'trainer'
                fighter['xp_received'] += gains[slot]
                fighter[source + '_xp_received'] += gains[slot]
                if gains[slot]:
                    episode['experience_events'].append(dict(step=event['step'], day=event['day'],
                        fighter_id=fid, slot=slot+1, amount=gains[slot], source=source,
                        opponent=army if source == 'battle' else None))
                if old[0][slot] and (new[0][slot] != old[0][slot] or new[1][slot] != old[1][slot]):
                    episode['upgrades'].append(dict(step=event['step'], day=event['day'],
                        fighter_id=fid, slot=slot+1,
                        kind='form' if new[0][slot] != old[0][slot] else 'level',
                        before=self._unit(old, slot, fid), after=unit,
                        awarded_xp=gains[slot], opponent=army))
                episode['final_party'].append({**unit, 'xp_received': fighter['xp_received'],
                    'battle_xp_received': fighter['battle_xp_received'],
                    'trainer_xp_received': fighter['trainer_xp_received']})
            episode['_slots'] = slots
            if event['battle_result']:
                episode['battles'].append(dict(step=event['step'], day=event['day'],
                    result=BATTLES[event['battle_result']], method=method, opponent=army))
            if event['defeated_enemy'] >= 0:
                episode['defeated_armies'].append(dict(step=event['step'], day=event['day'],
                    method=method, army=self._army(event['defeated_enemy'], enemy_levels)))
            self.event_count += 1
            if event['outcome']:
                episode['outcome'] = OUTCOMES[event['outcome']]
                episode['return'] = float(reward)
                episode['length'] = event['step']
                episode['end_position'] = [event['row'], event['column']]
                if episode['outcome'] == 'defeat':
                    episode['defeat'] = dict(cause='battle' if event['battle_result'] == 2 else 'other',
                        opponent=army if event['battle_result'] == 2 else None)
                episode.pop('_slots')
                lines.append(json.dumps(episode, ensure_ascii=False, separators=(',', ':')) + '\n')
                self._count(episode)
                del self.active[stream]
        if lines:
            # Compress already-transferred batches in memory. Only summaries
            # write to disk; learner/PPO blocks and GPU event collection stay unchanged.
            if self._gzip is None:
                self._gzip = gzip.GzipFile(fileobj=self._buffer, mode='wb', compresslevel=6, mtime=0)
            data = ''.join(lines).encode('utf-8')
            self._gzip.write(data)
            self._pending_records += len(lines)
            self.uncompressed_bytes += len(data)

    def _flush_episodes(self):
        """Append one complete gzip member; readers can inspect every saved prefix."""
        if not self._pending_records:
            return
        if self._gzip is not None:
            self._gzip.close()
            self._gzip = None
        with self.path.open('ab') as stream:
            stream.write(self._buffer.getvalue())
        self._buffer = io.BytesIO()
        self._pending_records = 0

    def _count(self, episode):
        self.completed += 1
        self.outcomes[episode['outcome']] += 1
        self.xp_total += sum(f['xp_received'] for f in episode['fighters'])
        formation = ' + '.join(f"{u['name']} ур. {u['level']}" for u in episode['final_party'])
        self.formations[formation or '(пустой отряд)'] += 1
        for unit in episode['final_party']:
            self.units[f"{unit['name']} ур. {unit['level']}"] += 1
        for victory in episode['defeated_armies']:
            self.defeated[(victory['army']['index'], victory['method'])] += 1
        if episode['defeat'] and episode['defeat']['opponent']:
            self.killers[episode['defeat']['opponent']['index']] += 1

    def summary(self):
        return dict(schema_version=SCHEMA_VERSION, completed_episodes=self.completed,
            pending_episodes=len(self.active), events=self.event_count, outcomes=dict(self.outcomes),
            episodes_file=self.path.name, episodes_file_bytes=self.path.stat().st_size,
            episodes_uncompressed_bytes=self.uncompressed_bytes,
            mean_awarded_xp=self.xp_total/max(1, self.completed),
            final_formations=[dict(formation=k, episodes=v) for k,v in self.formations.most_common()],
            final_units=[dict(unit=k, occurrences=v) for k,v in self.units.most_common()],
            defeated_armies=[dict(index=k[0], method=k[1], episodes=v)
                             for k,v in self.defeated.most_common()],
            defeat_opponents=[dict(index=k, episodes=v) for k,v in self.killers.most_common()])

    def write_summary(self, expected_completed=None, final=False):
        if expected_completed is not None and self.completed != expected_completed:
            raise ValueError(f'Episode report coverage mismatch: {self.completed} != {expected_completed}')
        self._flush_episodes()
        summary = self.summary()
        self._json('episode_reports.summary.json', summary)
        lines = ['# Отчёты по эпизодам', '',
            f"Карта: {self.game_map['name']}. Завершено эпизодов: {self.completed}.",
            f"Средний начисленный опыт всего отряда: {summary['mean_awarded_xp']:.2f}.", '',
            'Состав включает погибших бойцов; XP — до сброса при повышении и ограничения порогом.', '',
            '## Частые конечные составы', '', '| Состав | Эпизодов |', '|---|---:|']
        for entry in summary['final_formations'][:15]:
            lines.append(f"| {entry['formation'].replace('|', '/')} | {entry['episodes']} |")
        for title, entries in (('Побеждённые армии', summary['defeated_armies']),
                               ('Противники, завершившие эпизод поражением', summary['defeat_opponents'])):
            lines.extend(['', '## '+title, '', '| Отряд | Способ | Эпизодов |', '|---|---|---:|'])
            for entry in entries[:20]:
                roster = ', '.join(u['name'] for u in self.combat['enemies'][entry['index']] if u)
                lines.append(f"| №{entry['index']+1}: {roster.replace('|', '/')} | {entry.get('method', 'hero')} | {entry['episodes']} |")
        lines.extend(['', f'Полные составы, XP, последовательности улучшений и боёв: {self.path.name}.',
            f"Размер сжатого отчёта: {summary['episodes_file_bytes']} байт; "
            f"исходные JSON-записи: {self.uncompressed_bytes} байт.",
            'Незавершённые эпизоды не входят в статистику; partial-файл содержит состояние на последнем записанном событии.', ''])
        (self.output/'episode_reports.md').write_text('\n'.join(lines), encoding='utf-8')
        if final:
            self._json('episode_reports.partial.json', [
                {k:v for k,v in e.items() if k != '_slots'} for e in self.active.values()])
        return summary
