"""Replays a trace's neural decodes on the scopes of a hypothetical tensor core; see notes.md.

    python3 scopes.py output/<scene>.trace
"""
import sys

import numpy as np

# The decoder's (inputs, outputs) per layer, and `Site::Network` in trace.rs.
LAYERS = [(48, 64), (64, 48), (48, 32), (32, 16)]
NETWORK = 10
# Outputs M, samples N and inputs K per operation.
SCOPES = {'quad 8x4x8': (8, 4, 8), 'warp 16x16x16': (16, 16, 16), 'SM 128x256x16': (128, 256, 16)}
# Windows in warps: one SM holds 48 resident warps on Ada.
WINDOWS = {'CTA of 8 warps': 8, 'SM of 48 warps': 48, 'frame': 1 << 30}
# Time model: SM-cycles as the busiest SM resource, with latency hidden (weights double-buffered,
# pipelines by other warps). Ada's rates, measured: 4 warp instructions issue per cycle, the tensor
# pipe does 1024 int8 MACs, dp4a's fmaheavy pipe 2 cycles per warp instruction per sub-partition,
# and the L1/shared data path moves 128 bytes, a whole shuffle. Instructions per unit are from the
# Ada decoders' SASS. Weights reach tensor operands without instructions; scales and biases are
# 128-bit loads.
COMMON = 1390  # latent sampling and output finishing per warp decode, every scheme
TILE, IMMA = 531, 38  # per occupied 16-row tile
PASS = 15 + 20  # per warp pass: set loop, and each lane's scales and biases
TRANSPOSE, SHUFFLES = 201, 84  # fragment transposes per warp decode
# Per quad round: each thread activates its whole sample, two tiles' worth, with quad MMAs, the
# set loop and every scale and bias.
ROUND = 2 * (TILE - IMMA) + sum(o // 8 * (i // 8) for i, o in LAYERS) + 15 + 80
DP4A, HEAVY = 4111, 2184  # dp4a network per warp decode, of which on fmaheavy
SET = 8192 + 160 * 8  # weight, scale and bias bytes per set fetch
REGROUP, REGROUP_SHUFFLES = 30, 21  # in-warp lane permutation; the quad scope then moves inputs and outputs
SORT, SORT_BYTES = 90, 32 * 2 * 84  # per warp decode, sorting across an SM through shared memory
RECORD = np.dtype([('warp', '<u4'), ('pass', 'u1'), ('site', 'u1'), ('material', '<u4'), ('mask', '<u4')])

data = open(sys.argv[1], 'rb').read()
assert data[:4] == b'RTRC' and int.from_bytes(data[4:8], 'little') == 6, 'needs an RTRC version 6 trace'
# Skip the op table: per site, a u16 count per op type for its work and for its critical path.
sites, ops = (int.from_bytes(data[i:i + 4], 'little') for i in (20, 24))
records = np.frombuffer(data, RECORD, offset=28 + 4 * sites * ops)
records = records[records['site'] == NETWORK]
useful = sum(i * o for i, o in LAYERS)

def counts(r, window):
    """Samples per set within each window of `window` warps, ordered by window, and their windows."""
    key = (r['pass'].astype(np.int64) << 56) | (r['warp'].astype(np.int64) // window << 24) | r['material']
    keys, inverse = np.unique(key, return_inverse=True)
    return np.bincount(inverse, np.bitwise_count(r['mask'])).astype(np.int64), keys >> 24

def tiles(r, n, window):
    """n-sample tiles covering records r: each set's samples within windows of `window` warps,
    or for window 0 its occupied groups of n lanes."""
    if window == 0:
        return sum(int(np.count_nonzero(r['mask'] >> (n * g) & (1 << n) - 1)) for g in range(32 // n))
    return int(-(-counts(r, window)[0] // n).sum())

def packed(groups, sizes, unit):
    """Batches and (batch, item) pairs when each group's items, in order, fill batches of `unit`."""
    end = np.cumsum(sizes)
    first = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    end -= np.repeat(np.r_[0, end[first[1:] - 1]], np.diff(np.r_[first, len(sizes)]))
    last = np.r_[first[1:], len(sizes)] - 1
    return int(-(-end[last] // unit).sum()), int(((end - 1) // unit - (end - sizes) // unit + 1).sum())

def lane_groups(r):
    """Quad rounds and (round, set) pairs when each quad serves its own sets, one per round."""
    _, warp = np.unique((r['pass'].astype(np.int64) << 32) | r['warp'], return_inverse=True)
    record, quad = np.nonzero((r['mask'][:, None] >> 4 * np.arange(8) & 15) != 0)
    group = warp[record] * 8 + quad
    order = np.lexsort((r['material'][record], group))
    group, record = group[order], record[order]
    first = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    rank = np.arange(len(group)) - np.repeat(first, np.diff(np.r_[first, len(group)]))
    rounds = np.zeros(warp.max() + 1, np.int64)
    np.maximum.at(rounds, group // 8, rank + 1)
    pairs = np.unique((group // 8 << 37) | (rank.astype(np.int64) << 32) | r['material'][record])
    return int(rounds.sum()), len(pairs)

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
    # Instructions, pipe cycles per sub-partition, and data-path bytes for each scheme.
    d, sets = decodes, len(r)
    warp_counts, warps = counts(r, 1)
    sm_counts, windows = counts(r, 48)
    halves = tiles(r, 16, 0)
    regrouped = tiles(r, 16, 1)
    _, sorted_passes = packed(windows, sm_counts, 32)
    _, sorted_tiles = packed(windows, sm_counts, 16)
    rounds, round_sets = lane_groups(r)
    regrouped_rounds, regrouped_sets = packed(warps, -(-warp_counts // 4), 8)
    sorted_rounds, sorted_sets = packed(windows, -(-sm_counts // 4), 8)
    tile, quad_round = 16 * useful / 256, 32 * useful / 256
    schemes = {
        'SIMT dp4a': (d * (COMMON + DP4A), 2 * d * HEAVY, sets * SET),
        'warp as Ada runs it': (d * (COMMON + TRANSPOSE) + sets * PASS + halves * TILE, halves * tile,
                                d * SHUFFLES * 128 + sets * SET),
        'warp regrouped': (d * (COMMON + TRANSPOSE + REGROUP) + sets * PASS + regrouped * TILE, regrouped * tile,
                           d * SHUFFLES * 128 + sets * SET),
        'warp SM-sorted': (d * (COMMON + SORT) + sorted_passes * PASS + sorted_tiles * TILE, sorted_tiles * tile,
                           d * SORT_BYTES + sorted_passes * SET),
        'quad lane groups': (d * COMMON + rounds * ROUND, rounds * quad_round, round_sets * SET),
        'quad regrouped': (d * (COMMON + REGROUP + REGROUP_SHUFFLES) + regrouped_rounds * ROUND,
                           regrouped_rounds * quad_round, d * REGROUP_SHUFFLES * 128 + regrouped_sets * SET),
        'quad SM-sorted': (d * (COMMON + SORT) + sorted_rounds * ROUND, sorted_rounds * quad_round,
                           d * SORT_BYTES + sorted_sets * SET),
    }
    print(f"  {'time model, SM-cycles per warp decode':38s} {'issue':>6s} {'pipe':>6s} {'data':>6s} {'time':>6s} {'vs dp4a':>8s}")
    base = max(v / s for v, s in zip(schemes['SIMT dp4a'], (4, 4, 128)))
    for scheme, demand in schemes.items():
        cycles = [v / s / d for v, s in zip(demand, (4, 4, 128))]
        print(f'  {scheme:38s} ' + ' '.join(f'{c:6.0f}' for c in cycles) +
              f' {max(cycles):6.0f} {base / d / max(cycles):7.2f}x')
