"""Original landmarks block movement on Default and The Charmed Child.

Sources: docs/DEFAULT_MAP.md#ориентиры and docs/MAPS.md («Ориентиры S117»).
Reads the map JSON and the landmark extracts only; no environment or compilation.
"""
import json
from pathlib import Path

import pytest

from scripts.build_default_map import cell, landmark_cells

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.core
@pytest.mark.parametrize('name, count, tiles, corners', [
    # Default: the 5x5 landmark mountain in the south-west; S117: the Fall at (25, 13).
    ('default', 122, 287, [[44, 1], [48, 5]]),
    ('the_charmed_child', 27, 35, [[14, 26], [16, 28]]),
])
def test_landmarks_are_obstacles(name, count, tiles, corners):
    """Every landmark tile of the extract is an obstacle of the map and never terrain."""
    source = ROOT/'docs/maps'/name/'landmarks.json'
    game_map = json.loads((ROOT/'maps'/f'{name}.json').read_text(encoding='utf-8'))
    cells = {tuple(cell(p)) for p in landmark_cells(source)}
    obstacles = {tuple(p) for p in game_map['obstacles']}
    terrain = {tuple(p) for points in game_map['terrain'].values() for p in points}
    assert len(json.loads(source.read_text(encoding='utf-8'))['landmarks']) == count
    assert len(cells) == tiles and cells <= obstacles and not cells & terrain
    assert all(tuple(p) in obstacles for p in corners)
