"""Read source data without importing an environment or initializing JAX."""
import ast
import json
from pathlib import Path
import struct


def dbf(path):
    data = path.read_bytes()
    count, header, stride = struct.unpack_from('<IHH', data, 4)
    fields, offset = [], 1
    for cursor in range(32, header - 1, 32):
        if data[cursor] == 13:
            break
        name = data[cursor:cursor+11].split(b'\0')[0].decode('ascii')
        width = data[cursor+16]
        fields.append((name, offset, width))
        offset += width
    return [{name: data[header+i*stride+start:header+i*stride+start+width].decode('cp1251').strip()
             for name, start, width in fields}
            for i in range(count) if data[header+i*stride] != 42]


def reference_data(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'DATA' for target in node.targets))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    game = Path('C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves/Globals')
    tables = {p.stem.lower(): dbf(p) for p in game.iterdir() if p.suffix.lower() == '.dbf'}
    ref = reference_data(Path('C:/Disciples 2 python reference/Big_map/data_dicts_compact_lines.py'))
    output = root/'results/unit-catalog'
    output.mkdir(parents=True, exist_ok=True)
    (output/'sources.json').write_text(json.dumps(dict(tables=tables, reference=ref), ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ('gunits', 'gattacks', 'gdynupgr', 'gimmu', 'gimmuc', 'gtransf', 'latts', 'lattc', 'lattr', 'lunitc'):
        print(name, len(tables[name]), json.dumps(tables[name][:2], ensure_ascii=False))
    print('Reference units:', len(ref), 'types:', sorted({r['тип'] for r in ref}))
