"""Gives a scene's textured materials FlightHelmet's trained NTC sets as stand-ins, so any scene
can be traced for decode batching: textures decode wrong, but materials sharing their base
color, metallic-roughness and normal images share a set and every other material has its own.

    python3 standins.py SCENE [--remove]
"""
import json, pathlib, struct, sys

scenes = pathlib.Path(__file__).resolve().parent / 'assets/scenes'
folder = scenes / sys.argv[1]
out = folder / 'ntc'
assert folder.name != 'flight-helmet', 'FlightHelmet has trained sets'
if sys.argv[2:] == ['--remove']:
    for link in out.glob('*.ntc'):
        link.unlink()
    out.rmdir()
    sys.exit()

path = next(p for p in folder.iterdir() if p.suffix in ('.gltf', '.glb'))
blob = path.read_bytes()
g = json.loads(blob[20:20 + struct.unpack_from('<I', blob, 12)[0]] if path.suffix == '.glb' else blob)
sources = sorted(p for p in (scenes / 'flight-helmet/ntc').glob('*.ntc') if not p.is_symlink())
out.mkdir()
sets = {}
for index, m in enumerate(g['materials']):
    pbr = m.get('pbrMetallicRoughness', {})
    slots = [pbr.get('baseColorTexture'), pbr.get('metallicRoughnessTexture'), m.get('normalTexture')]
    images = tuple(g['textures'][s['index']].get('source') if s else None for s in slots)
    # The renderer names an unnamed material's set by its index.
    link = out / f"{m.get('name', index)}.ntc"
    if images == (None, None, None) or link.exists():
        continue
    if images in sets:
        # Same canonical file, so the loader shares the set.
        link.symlink_to(sets[images].name)
    else:
        # A hard link keeps its own canonical path, so the loader loads a distinct set.
        link.hardlink_to(sources[len(sets) % len(sources)])
        sets[images] = link
print(f'{out}: {len(sets)} stand-in sets')
