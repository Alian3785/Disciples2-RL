"""Pin S101 limits and Bethrezen's two leadership modifiers from installed DBFs.

Read-only game access; no game, IDA, JAX or training is run.
"""
import argparse
import hashlib
import json
from pathlib import Path

from inspect_unit_sources import dbf


def inspect(game):
    """Read S101 limits with platform-independent relative paths in the source manifest."""
    files = {p.relative_to(game).as_posix().lower(): p for folder in ('Scens', 'Globals')
             for p in (game/folder).iterdir() if p.suffix.lower() == '.dbf'}
    hashes = {}

    def read(name):
        """Read one DBF and retain its source digest under a portable path."""
        path = files[name]
        hashes[path.relative_to(game).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dbf(path)

    info = next(r for r in read('scens/sinfo.dbf') if r['NO_SCEN'] == 'S101SC0000')
    instance = next(r for r in read('scens/sunit.dbf') if r['UNIT_ID'] == 'S101UN0064')
    modifiers = [r for r in read('scens/sunitm.dbf') if r['UNIT_ID'] == instance['UNIT_ID']]
    profile = next(r for r in read('globals/gunits.dbf') if r['UNIT_ID'].lower() == instance['TYPE'].lower())
    ids = {r['MODIF_ID'].lower() for r in modifiers}
    definitions = [r for r in read('globals/gmodif.dbf') if r['MODIF_ID'].lower() in ids]
    effects = [r for r in read('globals/gmodifl.dbf') if r['BELONGS_TO'].lower() in ids]
    categories = read('globals/lmodife.dbf')
    assert next(r for r in categories if r['ID'] == '2')['TEXT'] == 'L_LEADERSHIP'
    delta = sum(int(e['NUMBER']) for m in modifiers for e in effects
                if e['BELONGS_TO'].lower() == m['MODIF_ID'].lower() and e['TYPE'] == '2')
    manual = game/'Scenedit manual/Disciples 2 Scenario Editor Manual.htm'
    hashes[manual.relative_to(game).as_posix()] = hashlib.sha256(manual.read_bytes()).hexdigest()
    return dict(scenario='S101SC0000', name=info['NAME'],
        scenario_limits={key: int(info[key]) for key in ('MAX_UNIT', 'MAX_LEADER', 'MAX_CITY', 'MAX_SPELL')},
        bethrezen_instance=instance, bethrezen_profile=profile, modifiers=modifiers,
        modifier_definitions=definitions, modifier_effects=effects,
        leadership=dict(base_internal=int(profile['LEADERSHIP']), scenario_delta=delta,
            effective_internal=int(profile['LEADERSHIP'])+delta,
            displayed_companion_capacity=int(profile['LEADERSHIP'])+delta-1,
            conversion_source='https://github.com/VladimirMakeev/D2ModdingToolset/blob/master/mss32/src/enclayoutunithooks.cpp#L652-L658'),
        semantics=dict(MAX_UNIT='Highest unit upgrade building tier, not a numerical unit level cap.',
            MAX_LEADER='Leader promotion ceiling; higher imported levels are preserved without further promotion.',
            MAX_SPELL='Highest spell level learnable in the Mage Guild; not a ban on scripted spells or items.'),
        sha256=hashes, original_game_executed=False, ida_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/
                        'docs/maps/bethrezens_retreat/restrictions.json')
    args = parser.parse_args()
    report = inspect(args.game)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(limits=report['scenario_limits'], leadership=report['leadership']), ensure_ascii=False))


if __name__ == '__main__':
    main()
