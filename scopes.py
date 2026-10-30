"""Whole-shader cycles per warp of a trace on the scopes of a hypothetical tensor core; see notes.md.

    python3 scopes.py output/<scene>.trace ...

Every traced region counts. A per-lane region costs a warp's max(issue, SFU pipe, critical path)
once per pass, whatever its lanes' materials; a network region costs what each scheme spends on
the warp's texture sets or materials. Traversal is untraced, so excluded. An NTC trace reports
its decode ('ntc'); a neural-material trace reports its networks ('neural') and neural materials
whose latents are NTC-compressed, each point running the sampler also decoding its material's NTC
set ('both'). Prints
each bounce depth, then the frame: every pass, summed per warp.
"""
import functools, pathlib, sys

import numpy as np

RECORD = np.dtype([('warp', '<u4'), ('pass', 'u1'), ('site', 'u1'), ('material', '<u4'), ('mask', '<u4')])
# `Site` numbers in trace.rs.
LATENTS, NETWORK, APPEARANCE, NN, SCATTER, LIGHT = 9, 10, 11, 12, 13, 14
# Latency of a load, store, FP32, SFU and integer op; SFU cycles per warp instruction; and the one
# in-warp shuffle that regroups lanes before a network.
LAT = np.array([30, 1, 4, 15, 4]); SFU_PIPE = 8; SHUFFLE = 100
# Cycles per m16n8k32 int8 or m16n8k16 FP16 warp MMA; a quad MMA (8 quads of 8x4x8) takes the same
# per MAC. A warp's MMAs serialize.
MMA = 20
# The neural material's FP16 multiplies, (outputs, inputs) per op column after the int8 ones.
SHAPES = [(12, 8), (16, 11), (16, 16), (9, 16), (16, 24), (3, 16)]


def load(path):
    d = open(path, 'rb').read()
    assert d[:4] == b'RTRC' and int.from_bytes(d[4:8], 'little') == 9, 'needs an RTRC version 9 trace'
    w, h, passes, bounces, sites, ops = (int(x) for x in np.frombuffer(d, '<u4', 6, 8))
    # Per site, a u16 count per op type for its work and for its critical path.
    table = np.frombuffer(d, '<u2', 2 * sites * ops, 32).reshape(sites, 2, ops).astype(int)
    return bounces, table, np.frombuffer(d, RECORD, offset=32 + 4 * sites * ops)


def lane_cost(table, s):
    work, path = table[s]
    return max(work[:5].sum(), work[3] * SFU_PIPE, (path[:5] * LAT).sum())


def ntc_model(table):
    """Warp, quad and SIMT cycles of the NTC network, 48-64-48-32-16 in int8."""
    work, path = table[NETWORK]
    issue, p = work[:5].sum(), (path[:5] * LAT).sum()
    # t occupied 16-row tiles: 16 samples' activations spread over the warp, and per tile 38 MMA
    # instructions with 32 m16n8k32s' MACs.
    warp = lambda t: max(issue * t / 2 + 38 * t, 32 * MMA * t, p + 32 * MMA * t)
    # A round: every lane activates its own sample, and 128 quad MMAs.
    quad = lambda r: r * max(issue + 128, 64 * MMA, p + 64 * MMA)
    # Each lane alone, among k sets: 2048 dp4a at 2 cycles and 512 16-byte weight loads, which with
    # the scale and bias loads fill 4 L1 cycles each, or one per set beyond 4. Path: a layer's
    # weight load, then its dot products' 48 dependent dp4a.
    simt = lambda k: max(issue + 2048 + 512, 2 * 2048, (512 + work[0]) * max(4, k),
                         4 * LAT[0] + 48 * LAT[4] + p)
    return warp, quad, simt


def nn_model(table, site):
    """Warp, quad and SIMT cycles of a neural material's networks in a region, in FP16."""
    work, path = table[site]
    counts, chain = work[9:15], path[9:15]
    mmas = lambda ops, k: sum(c * -(-o // 8) * -(-i // k) for c, (o, i) in zip(ops, SHAPES))
    # m16n8k16 per 16-row tile, quad MMAs per round.
    wi, wc, qi, qc = mmas(counts, 16), mmas(chain, 16), mmas(counts, 8), mmas(chain, 8)
    issue, p = work[:5].sum(), (path[:5] * LAT).sum()
    # SIMT: HFMA2 does 2 MACs and a 16-byte load carries 8 weights; each layer on the path loads
    # its weights, then chains K / 2 HFMA2.
    macs = sum(c * o * i for c, (o, i) in zip(counts, SHAPES))
    loads = -(-macs // 8)
    sp = p + sum(c * (LAT[0] + -(-i // 2) * LAT[4]) for c, (o, i) in zip(chain, SHAPES))
    warp = lambda t: max(issue * t / 2 + wi * t, wi * MMA * t, p + wc * MMA + (t - 1) * wi * MMA)
    quad = lambda r: r * max(issue + qi, qi * MMA, p + qc * MMA)
    simt = lambda k: max(issue + macs // 2 + loads, (loads + work[0]) * max(4, k), sp)
    return warp, quad, simt


def fits(cs, rounds, descending, pick):
    """Whether lane counts cs fit 8 quads of 4 lanes, each quad serving at most `rounds` sets."""
    free, items = [4] * 8, [0] * 8
    for c in sorted(cs, reverse=descending):
        while c:
            cand = [i for i in range(8) if items[i] < rounds and free[i] > 0]
            if not cand:
                return False
            i = pick(cand, free, c)
            take = min(c, free[i])
            free[i] -= take
            items[i] += 1
            c -= take
    return True


PICKS = [
    lambda cand, f, c: min(cand, key=lambda i: (f[i] < c, f[i] if f[i] >= c else -f[i])),  # best fit, else largest
    lambda cand, f, c: max(cand, key=lambda i: f[i]),  # worst fit
    lambda cand, f, c: min(cand, key=lambda i: (f[i] != c, f[i] < min(c, 4), -f[i])),  # exact, else whole quad, else largest
]


@functools.lru_cache(None)
def shuffled_rounds(cs):
    """Fewest quad rounds the greedy packings reach after one in-warp shuffle."""
    lo = max(1, -(-sum(-(-c // 4) for c in cs) // 8))
    return next(r for r in range(lo, 9) if any(fits(cs, r, o, p) for o in (True, False) for p in PICKS))


def schemes(masks, model):
    """[warp, warp+shuffle, quad, quad+shuffle, SIMT] network cycles for one region's masks."""
    warp, quad, simt = model
    cs = tuple(sorted(bin(x).count('1') for x in masks))
    return np.array([sum(warp((x & 0xFFFF != 0) + (x >> 16 != 0)) for x in masks),
                     sum(warp(-(-c // 16)) for c in cs),
                     quad(max(sum(1 for x in masks if x >> 4 * q & 15) for q in range(8))),
                     quad(shuffled_rounds(cs)), simt(len(masks))], float)


def analyze(bounces, table, rec, mode):
    """Per depth (0 primary), per warp pass: common and hot (latents and networks only) per-lane
    cycles, then each scheme's network cycles."""
    nets = {NETWORK: ntc_model(table)} if mode == 'ntc' else {s: nn_model(table, s) for s in (NN, SCATTER, LIGHT)}
    ntc = ntc_model(table)
    lane = {s: lane_cost(table, s) for s in range(len(table))}
    rec = rec[np.lexsort((rec['site'], rec['warp'], rec['pass']))]
    key = (rec['pass'].astype(np.int64) << 32) | rec['warp']
    first = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(rec)]
    out = {}
    for a, b in zip(first[:-1], first[1:]):
        g = rec[a:b]
        # Pass 1 + r b + d is bounce ray r at depth d.
        p = int(g['pass'][0]); depth = 0 if p == 0 else 1 + (p - 1) % bounces
        sites = set(int(s) for s in g['site'])
        common = sum(lane[s] for s in sites if s not in nets)
        hot = lane[LATENTS] * (LATENTS in sites) if mode == 'ntc' else lane[APPEARANCE] * (APPEARANCE in sites)
        net = np.zeros(5)
        for s, model in nets.items():
            m = [int(x) for x in g['mask'][g['site'] == s]]
            if m:
                net += schemes(m, model)
        if mode == 'both':
            m = [int(x) for x in g['mask'][g['site'] == NN]]
            if m:
                net += schemes(m, ntc)
                common += lane[LATENTS]; hot += lane[LATENTS]
        if net.any():
            net[[1, 3]] += SHUFFLE
        out.setdefault(depth, []).append((common, hot, *net))
    return out


def report(label, x):
    """Per-warp choices: the warp scope shuffles only when it helps; today's hybrid may also run
    SIMT; the flexible core picks the warp or quad scope, either shuffled or not."""
    def costs(base, net):
        warp = base + np.minimum(net[:, 0], net[:, 1])
        quad = base + np.minimum(net[:, 2], net[:, 3])
        simt = base + net[:, 4]
        hybrid = np.minimum(warp, simt)
        flexible = np.minimum(warp, quad)
        return [v.mean() for v in (warp, quad, simt, hybrid, flexible)]
    net = x[:, 2:]
    W, H = costs(x[:, 0], net), costs(x[:, 1], net)
    f = lambda v: f'{v:4.2f}x'
    print(f'  {label:8s} ' + ' '.join(f'{v:6.0f}' for v in W) +
          f' | quad/warp {f(W[0] / W[1])}  quad/SIMT {f(W[2] / W[1])}  flexible/hybrid {f(W[3] / W[4])}'
          f'   [region only: {f(H[0] / H[1])} {f(H[2] / H[1])} {f(H[3] / H[4])}]')


for path in sys.argv[1:]:
    bounces, table, records = load(path)
    for mode in ('neural', 'both') if (records['site'] == NN).any() else ('ntc',):
        out = analyze(bounces, table, records, mode)
        print(f'{pathlib.Path(path).stem} [{mode}]  whole-shader cycles/warp: warp quad SIMT hybrid flexible')
        for depth in sorted(out):
            report('primary' if depth == 0 else f'bounce {depth}', np.array(out[depth], float))
        # Every pass, per warp position: passes summed, warps averaged.
        rows = sum(map(len, out.values()))
        report('frame', np.concatenate([np.array(v, float) for v in out.values()]) * (rows / len(out[0])))
