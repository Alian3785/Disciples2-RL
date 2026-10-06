"""Documentation assets reflect actual reset state without shadowing map modules."""
import importlib
import json
from pathlib import Path

from PIL import Image
import pytest

from maps import available_maps
from tools.generate_map_layouts import extract, roster_entries

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', available_maps())
def test_map_layout_matches_runtime(name):
    folder = ROOT / 'maps' / name
    actual = json.loads((folder / 'layout.json').read_text(encoding='utf-8'))
    expected = json.loads(json.dumps(extract(name), ensure_ascii=False))
    assert actual == expected
    assert not (folder / '__init__.py').exists()
    assert Path(importlib.import_module('maps.' + name).__file__).name == name + '.py'
    assert (folder / 'README.md').is_file()
    with Image.open(folder / 'layout.png') as im:
        im.verify()
    if any(e['wave'] for e in actual['enemies']) or actual['conditional_assaults']:
        with Image.open(folder / 'waves.png') as im:
            im.verify()


def test_mirrow_legend_covers_both_halves_and_tiamat():
    data = extract('mirrow_match')
    assert len(data['enemies']) == 39
    assert len(roster_entries(data)) == 20
    assert sum(e['marker'] == '★' for e in data['enemies']) == 1
    assert all(sum(e['marker'] == str(i) for e in data['enemies']) == 2 for i in range(1,20))


def test_dormant_waves_are_not_initial_armies():
    data = extract('super_last_stand')
    assert sum(e['alive'] for e in data['enemies']) == 7
    future = [e for e in data['enemies'] if e['wave']]
    assert len(future) == 33
    assert not any(e['alive'] for e in future)
    assert sorted(e['wave']['spawn_turn'] for e in future) == list(range(5,166,5))
    assert len({tuple(e['position']) for e in future}) == 3


def test_shared_settlement_armies_are_preserved():
    data = extract('wotans_retribution')
    capital = [e for e in data['enemies'] if e['id'] in (5,10)]
    assert len(capital) == 2
    assert capital[0]['position'] == capital[1]['position'] == [13,33]
    assert all(e['formation_cells'] <= 6 for e in data['enemies'])
    assert all(e['fighter_count'] == len(e['units']) for e in data['enemies'])


def test_small_uses_runtime_guardian_coordinates():
    data = extract('small')
    positions = {e['id']:e['position'] for e in data['enemies']}
    assert data['grid_size'] == 24
    assert positions[70] == [8,5]
    assert positions[74] == [22,18]
    assert positions[34] == positions[68] == [14,16]
    assert positions[35] == positions[69] == [20,12]
