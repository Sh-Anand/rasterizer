"""Replays a trace's neural decodes on the proposed Radiance tensor-core scopes; see notes.md.

    python3 scopes.py output/<scene>.trace
"""
import sys

import numpy as np

# The decoder's (inputs, outputs) per layer, and `Site::Neural` in trace.rs.
LAYERS = [(48, 64), (64, 48), (48, 32), (32, 16)]
NEURAL = 9
# Outputs M, samples N and inputs K per operation.
SCOPES = {'quad 8x4x8': (8, 4, 8), 'warp 16x16x16': (16, 16, 16), 'SM 128x256x16': (128, 256, 16)}
# Windows in warps: one SM holds 48 resident warps on Ada.
WINDOWS = {'CTA of 8 warps': 8, 'SM of 48 warps': 48, 'frame': 1 << 30}
RECORD = np.dtype([('warp', '<u4'), ('pass', 'u1'), ('site', 'u1'), ('material', '<u4'),
                   ('mask', '<u4'), ('ops', '<u2', 5)])

data = open(sys.argv[1], 'rb').read()
assert data[:4] == b'RTRC' and int.from_bytes(data[4:8], 'little') == 5, 'needs an RTRC version 5 trace'
records = np.frombuffer(data, RECORD, offset=20)
records = records[records['site'] == NEURAL]
useful = sum(i * o for i, o in LAYERS)

def tiles(r, n, window):
    """n-sample tiles covering records r: each set's samples within windows of `window` warps,
    or for window 0 its occupied groups of n lanes."""
    if window == 0:
        return sum(int(np.count_nonzero(r['mask'] >> (n * g) & (1 << n) - 1)) for g in range(32 // n))
    key = (r['pass'].astype(np.int64) << 56) | (r['warp'].astype(np.int64) // window << 24) | r['material']
    _, inverse = np.unique(key, return_inverse=True)
    return int(np.ceil(np.bincount(inverse, np.bitwise_count(r['mask'])) / n).sum())

for name, kind in [('primary', records['pass'] == 0), ('bounce', records['pass'] > 0)]:
    r = records[kind]
    if not len(r):
        continue
    samples = int(np.bitwise_count(r['mask']).sum())
    key = (r['pass'].astype(np.int64) << 32) | r['warp']
    decodes = len(np.unique(key))
    print(f'{name}: {samples} decoding samples in {decodes} warp decodes, '
          f'{len(r) / decodes:.2f} sets per warp decode; SIMT lanes busy {100 * samples / (32 * decodes):.1f}%')
    print(f"  {'scope':15s} {'tiling':16s} {'operations':>12s} {'useful MACs':>12s} {'weight B/sample':>16s}  data movement")
    for scope, (m, n, k) in SCOPES.items():
        per_tile = sum(-(-o // m) * -(-i // k) for i, o in LAYERS)
        for tiling, window in [('lane groups', 0), ('warp regrouped', 1)] + list(WINDOWS.items()):
            if n > 32 and window in (0, 1):
                continue
            t = tiles(r, n, window)
            moved = 'memory' if window > 1 else 'none' if n == 4 and window == 0 else 'shuffles'
            print(f'  {scope:15s} {tiling:16s} {t * per_tile:12d} {100 * samples * useful / (t * per_tile * m * n * k):11.1f}%'
                  f' {t * useful / samples:16.0f}  {moved}')
