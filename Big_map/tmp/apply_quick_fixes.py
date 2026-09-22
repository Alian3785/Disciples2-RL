from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
backup = ROOT/'tmp/quick_fixes_before'
backup.mkdir(exist_ok=True)

def edit(name, transform):
    path = ROOT/name
    raw = path.read_bytes()
    assert not (backup/name).exists(), name
    (backup/name).write_bytes(raw)
    text = raw.decode('utf8')
    updated = transform(text)
    assert updated != text, name
    path.write_bytes(updated.encode('utf8'))

def units(text):
    for name, field, old, new in [('Дева рощи','здоровье',75,85),('Сильфида','нужный опыт',2535,2470),('Мизраэль','инит',95,90)]:
        pattern = r'(?m)^.*"кто":"'+name+r'".*$'
        match = re.search(pattern,text)
        assert match and f'"{field}":{old}' in match[0]
        text = text[:match.start()]+match[0].replace(f'"{field}":{old}',f'"{field}":{new}',1)+text[match.end():]
    return text

def scrolls(text):
    for old,new in [('Хуорн','Темный энт'),('Энт Малый','Малый энт'),('Энт Большой','Великий энт'),('Вердант','Буйный энт')]:
        before=f"'summon_unit_name': '{old}'"
        assert text.count(before)==1
        text=text.replace(before,f"'summon_unit_name': '{new}'")
    return text

def potions(text):
    nl='\r\n' if '\r\n' in text else '\n'
    aliases=['Potion of Might','Might Potion','Зелье мощи']
    for name in aliases:
        line=f'                "{name}",'+nl
        assert text.count(line)==1
        text=text.replace(line,'')
    anchor='                "Potion of Energy",'+nl
    assert text.count(anchor)==1
    return text.replace(anchor, ''.join(f'                "{name}",'+nl for name in aliases)+anchor)

def spells(text):
    updates={
        'mcl_d2_s015':{'life':(150,225),'rune':(150,225)},
        'mcl_d2_s020':{'life':(400,200),'death':(400,200),'rune':(800,400)},
        'lod_d2_s016':{'hell':(100,200),'life':(200,100)},
        'lod_d2_s017':{'hell':(100,200),'life':(200,100)},
        'lod_d2_s020':{'life':(150,0),'rune':(0,150)},
    }
    for key, fields in updates.items():
        match=re.search(r'    "'+key+r'": \{.*?\n    \},',text,re.S)
        assert match, key
        block=match[0]
        for field,(old,new) in fields.items():
            token=f'"use_{field}": {old},'
            assert block.count(token)==1,(key,field)
            block=block.replace(token,f'"use_{field}": {new},')
        text=text[:match.start()]+block+text[match.end():]
    return text

edit('data_dicts_compact_lines.py',units)
edit('scroll_spell_data.py',scrolls)
edit('campaign_env_data.py',potions)
edit('spells.py',spells)
print('Updated 3 unit fields, 4 summon references, Might aliases, 11 casting cost components.')
