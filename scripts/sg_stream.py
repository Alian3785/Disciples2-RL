"""Read the objects of a Disciples II .sg scenario stream without running the game.

Every object is WHAT + its RTTI class name, OBJ_ID, BEGOBJECT, fields and ENDOBJECT.
A field is its bare name followed by its value: a string (uint32 length including the
trailing zero, then the bytes), an int32 or a one-byte bool. Nothing separates fields,
so the candidate types come from the Scens DBF headers, which name the same columns,
and a candidate is accepted only when the next field name starts right after it.
An object id used as a field name holds the int32 count of the repeated field that
follows (inventory items, unit modifiers, ...); it is returned as '#<id>'.

Map blocks keep their 128 raw bytes as hex. Objects that do not parse are returned with an
'error' instead of fields. Event fields come from EVENT_FIELDS: a CMidEvent is its header,
COND_QTY conditions and EFFECT_QTY effects, each opened by its CATEGORY.
"""
import collections
import re
import struct
from pathlib import Path

RUN = re.compile(rb'[A-Z][A-Z0-9_]*')
COUNT = re.compile(rb'S\d{3}[A-Z]{2}[0-9a-f]{4}')
CLASS = re.compile(rb'\.\?AV(\w+)@@\x00')
BINARY = (b'BLOCKDATA', b'FOG')
# Stream-only names: inline texts and lists that the DBF tables keep elsewhere.
STREAM_STRINGS = ('WHAT OBJ_ID NAME_TXT DESC_TXT TXT_TITLE TXT_DESC TITLE NAME DESC BRIEFING '
                  'BRIEFLONG1 BRIEFLONG2 BRIEFLONG3 BRIEFLONG4 BRIEFLONG5 DEBUNKW DEBUNKW2 DEBUNKW3 '
                  'DEBUNKW4 DEBUNKW5 DEBUNKL DEBUNKLB CREATOR MODIF_ID ITEM_ID UNIT_ID SPELL_ID '
                  'BUILD_ID INFO_ID CAMPAIGN').split()
# CMidEvent condition/effect fields: no Scens DBF names them for custom .sg maps.
EVENT_FIELDS = dict(
    s=('ID_STKTEMP ID_LOC POPUP_TXT MUSIC SOUND IMAGE IMAGE2 ID_EVENT ID_PLAYER1 ID_PLAYER2 ID_STACK '
       'ORDER_TARG ID_ITEM ID_CITY ID_RUIN ID_UNIT ID_SITE ID_LMARK ID_SPELL TYPE_SPELL ID_MOUNT').split(),
    i='CHANCE ORDER COND_QTY CATEGORY FREQUENCY EFFECT_QTY NUM POPUP_SHOW MISC_INT PCT_LAND'.split(),
    b=('VERHUMAN VERDWARF VERUNDEAD VERHERETIC VERNEUTRAL VERELF ENABLED OCCUR_ONCE LEFT_SIDE AFF '
       'BOOLVALUE ENABLE WIN_SCEN FIRST_ONLY').split())


def dbf_types(scens):
    """Field name -> candidate kinds ('s', 'i', 'b') over every Scens DBF header."""
    types = collections.defaultdict(list)
    for path in sorted(Path(scens).iterdir()):
        if path.suffix.lower() != '.dbf':
            continue
        data = path.read_bytes()
        header = struct.unpack_from('<H', data, 8)[0]
        for cursor in range(32, header-1, 32):
            if data[cursor] == 13:
                break
            name = data[cursor:cursor+11].split(b'\0')[0]
            kind = {'C': 's', 'N': 'i', 'L': 'b'}.get(chr(data[cursor+11]), 's')
            if kind not in types[name]:
                types[name].append(kind)
    for name in STREAM_STRINGS:
        key = name.encode()
        types[key] = ['s']+[k for k in types.get(key, []) if k != 's']
    for kind, names in EVENT_FIELDS.items():
        for name in names:
            key = name.encode()
            types[key] = [kind]+[k for k in types.get(key, []) if k != kind]
    types[b'NUMBER'] = ['i']
    return dict(types)


class Reader:
    def __init__(self, scens):
        self.types = dbf_types(scens)

    def _starts_field(self, chunk, j):
        if j == len(chunk) or chunk.startswith(b'BEGOBJECT\0', j) or COUNT.match(chunk, j):
            return True
        m = RUN.match(chunk, j)
        return bool(m) and any(m.group()[:k] in self.types for k in range(1, len(m.group())+1))

    @staticmethod
    def _value(chunk, j, kind, binary):
        if kind == 's':
            if j+4 > len(chunk):
                return None
            n = struct.unpack_from('<I', chunk, j)[0]
            if not 0 < n <= len(chunk)-j-4:
                return None
            raw = chunk[j+4:j+4+n]
            if raw[-1:] == b'\0' and all(b >= 0x20 or b in (9, 10, 13) for b in raw[:-1]):
                return raw.rstrip(b'\0').decode('cp1251'), j+4+n
            return (raw.hex(), j+4+n) if binary else None
        if kind == 'i':
            return (struct.unpack_from('<i', chunk, j)[0], j+4) if j+4 <= len(chunk) else None
        return (chunk[j], j+1) if j < len(chunk) else None

    def fields(self, chunk):
        result, i = [], 0
        while i < len(chunk):
            if chunk.startswith(b'BEGOBJECT\0', i):
                i += len(b'BEGOBJECT\0')
                continue
            m = COUNT.match(chunk, i)
            if m:
                value, i = self._value(chunk, m.end(), 'i', False)
                result.append(['#'+m.group().decode(), value])
                continue
            m = RUN.match(chunk, i)
            if not m:
                raise ValueError(f'no field name at {i}')
            run = m.group()
            prefixes = [run[:k] for k in range(len(run), 0, -1)]
            candidates = [(p, k) for p in prefixes if p in self.types for k in self.types[p]]
            candidates += [(p, k) for p in prefixes if p in self.types for k in 'sib' if k not in self.types[p]]
            candidates += [(run, k) for k in 'sib']
            for name, kind in candidates:
                found = self._value(chunk, i+len(name), kind, name in BINARY)
                if found and self._starts_field(chunk, found[1]):
                    result.append([name.decode(), found[0]])
                    i = found[1]
                    break
            else:
                raise ValueError(f'cannot read {run!r} at {i}')
        return result

    def objects(self, data):
        found = []
        for m in CLASS.finditer(data):
            end = data.find(b'ENDOBJECT', m.end())
            entry = dict(cls=m.group(1).decode())
            try:
                entry['fields'] = self.fields(data[m.end():end])
            except ValueError as error:
                entry['error'] = str(error)
            found.append(entry)
        return found


class Obj:
    """Field access for one parsed object; repeated names keep their order."""

    def __init__(self, entry):
        self.cls = entry['cls']
        self.pairs = entry.get('fields', [])

    def __getitem__(self, name):
        return next(v for k, v in self.pairs if k == name)

    def get(self, name, default=None):
        return next((v for k, v in self.pairs if k == name), default)

    def all(self, name):
        return [v for k, v in self.pairs if k == name]

    @property
    def id(self):
        return self['OBJ_ID']
