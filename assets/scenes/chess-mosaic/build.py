"""Builds chess-mosaic.gltf: ChessCube's chess set in a closed box whose six faces are tiled with
panels, each using one of Sponza's full texture sets, so every surface carries a texture set.

    python3 assets/scenes/chess-mosaic/build.py [--materials M] [--panels N] [--out DIR]

M (1-24) sets how many distinct panel materials appear and N the panels per face edge; together
they sweep material coherence with geometry and lighting fixed.
"""
import argparse, io, json, pathlib, struct
from PIL import Image

here = pathlib.Path(__file__).resolve().parent
scenes = here.parent
parser = argparse.ArgumentParser()
parser.add_argument('--materials', type=int, default=24)
parser.add_argument('--panels', type=int, default=4)
parser.add_argument('--out', type=pathlib.Path, default=here)
args = parser.parse_args()

blob = (scenes / 'chess-cube/ChessCube.glb').read_bytes()
json_length = struct.unpack_from('<I', blob, 12)[0]
g = json.loads(blob[20:20 + json_length])
source = blob[20 + json_length + 8:20 + json_length + 8 + g['buffers'][0]['byteLength']]

# Halve the chess set's 2048^2 images: as RGBA32F mips beside Sponza's they need 4.3 GiB.
image_views = {image['bufferView'] for image in g['images']}
data = bytearray()
for i, view in enumerate(g['bufferViews']):
    payload = source[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
    if i in image_views:
        image = Image.open(io.BytesIO(payload))
        encoded = io.BytesIO()
        image.resize((image.width // 2, image.height // 2), Image.LANCZOS).save(encoded, image.format, quality=95)
        payload = encoded.getvalue()
    while len(data) % 4:
        data.append(0)
    view.update(byteOffset=len(data), byteLength=len(payload))
    data.extend(payload)

# Drop the Cornell box and its emitter; the sun is the renderer's only light.
cube = next(i for i, n in enumerate(g['nodes']) if n.get('name') == 'Cube')
roots = g['scenes'][g.get('scene', 0)]['nodes']
roots.remove(cube)
g['nodes'][cube] = {'name': 'RemovedCube'}

# Copy Sponza's materials that have base color, normal and metallic-roughness textures.
sponza = json.loads((scenes / 'sponza/Sponza.gltf').read_text())
sampler = len(g.setdefault('samplers', []))
g['samplers'].append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497})
remapped = {}
def texture(info):
    image = sponza['textures'][info['index']]['source']
    if image not in remapped:
        g.setdefault('images', []).append({'uri': '../sponza/' + sponza['images'][image]['uri']})
        g.setdefault('textures', []).append({'source': len(g['images']) - 1, 'sampler': sampler})
        remapped[image] = len(g['textures']) - 1
    return dict(info, index=remapped[image])
panel_materials = []
for m in sponza['materials']:
    pbr = m.get('pbrMetallicRoughness', {})
    if not {'baseColorTexture', 'metallicRoughnessTexture'} <= pbr.keys() or 'normalTexture' not in m:
        continue
    material = {
        'name': 'Panel_' + m.get('name', str(len(panel_materials))),
        'pbrMetallicRoughness': dict(pbr, baseColorTexture=texture(pbr['baseColorTexture']),
                                     metallicRoughnessTexture=texture(pbr['metallicRoughnessTexture'])),
        'normalTexture': texture(m['normalTexture']),
    }
    g['materials'].append(material)
    panel_materials.append(len(g['materials']) - 1)
assert 1 <= args.materials <= len(panel_materials), f'--materials must be 1..{len(panel_materials)}'
panel_materials = panel_materials[:args.materials]

# Box faces as (origin, edge u, edge v) with u x v pointing inward and v up the walls; matches
# the Cornell box. Single-sided panels face inward, so the shadow pass culls the ceiling and the
# sun lights the interior while bounce rays stay inside.
lo, hi = (-1.86, -0.058, -1.86), (1.86, 2.144, 1.86)
size = [h - l for l, h in zip(lo, hi)]
def edge(axis, length):
    return tuple(length if i == axis else 0.0 for i in range(3))
faces = [
    ((lo[0], lo[1], hi[2]), edge(2, -size[2]), edge(1, size[1])),  # x = lo, normal +x
    ((hi[0], lo[1], lo[2]), edge(2, size[2]), edge(1, size[1])),   # x = hi, normal -x
    ((lo[0], lo[1], hi[2]), edge(0, size[0]), edge(2, -size[2])),  # floor, normal +y
    ((lo[0], hi[1], lo[2]), edge(0, size[0]), edge(2, size[2])),   # ceiling, normal -y
    ((lo[0], lo[1], lo[2]), edge(0, size[0]), edge(1, size[1])),   # z = lo, normal +z
    ((hi[0], lo[1], hi[2]), edge(0, -size[0]), edge(1, size[1])),  # z = hi, normal -z
]
def add(a, b, s=1.0):
    return tuple(x + y * s for x, y in zip(a, b))
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def unit(a):
    n = sum(x * x for x in a) ** 0.5
    return tuple(x / n for x in a)

groups = {}
n = args.panels
for f, (origin, u, v) in enumerate(faces):
    normal = unit(cross(u, v))
    for i in range(n):
        for j in range(n):
            # A stride coprime with most counts scatters materials across neighbors.
            material = panel_materials[((f * n + i) * n + j) * 7 % len(panel_materials)]
            corner = add(add(origin, u, i / n), v, j / n)
            quad = [corner, add(corner, u, 1 / n), add(add(corner, u, 1 / n), v, 1 / n), add(corner, v, 1 / n)]
            groups.setdefault(material, []).append((quad, normal))

def append(payload, target=None):
    while len(data) % 4:
        data.append(0)
    view = {'buffer': 0, 'byteOffset': len(data), 'byteLength': len(payload)}
    if target:
        view['target'] = target
    data.extend(payload)
    g['bufferViews'].append(view)
    return len(g['bufferViews']) - 1
def accessor(values, kind, count, extra=None):
    g['accessors'].append(dict({'bufferView': values, 'componentType': 5126 if kind != 'SCALAR' else 5125,
                                'count': count, 'type': kind}, **(extra or {})))
    return len(g['accessors']) - 1

primitives = []
for material, quads in groups.items():
    positions, normals, uvs, indices = [], [], [], []
    for quad, normal in quads:
        base = len(positions)
        positions += quad
        normals += [normal] * 4
        uvs += [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        # The first edge's winding follows u, then v, so front faces point inward.
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    pmin = [min(p[k] for p in positions) for k in range(3)]
    pmax = [max(p[k] for p in positions) for k in range(3)]
    p = accessor(append(struct.pack(f'<{3 * len(positions)}f', *sum(positions, ())), 34962), 'VEC3', len(positions), {'min': pmin, 'max': pmax})
    nrm = accessor(append(struct.pack(f'<{3 * len(normals)}f', *sum(normals, ())), 34962), 'VEC3', len(normals))
    uv = accessor(append(struct.pack(f'<{2 * len(uvs)}f', *sum(uvs, ())), 34962), 'VEC2', len(uvs))
    idx = accessor(append(struct.pack(f'<{len(indices)}I', *indices), 34963), 'SCALAR', len(indices))
    primitives.append({'attributes': {'POSITION': p, 'NORMAL': nrm, 'TEXCOORD_0': uv}, 'indices': idx, 'material': material})
g['meshes'].append({'name': 'Mosaic', 'primitives': primitives})
g['nodes'].append({'name': 'Mosaic', 'mesh': len(g['meshes']) - 1})
roots.append(len(g['nodes']) - 1)

while len(data) % 4:
    data.append(0)
args.out.mkdir(parents=True, exist_ok=True)
name = 'chess-mosaic'
g['buffers'] = [{'uri': f'{name}.bin', 'byteLength': len(data)}]
(args.out / f'{name}.bin').write_bytes(data)
(args.out / f'{name}.gltf').write_text(json.dumps(g))
print(f'{args.out / name}.gltf: {len(panel_materials)} panel materials, {6 * n * n} panels, {len(g["materials"])} materials')
