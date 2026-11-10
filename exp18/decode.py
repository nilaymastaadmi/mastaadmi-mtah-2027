"""EXPERIMENT-18 change-aware Viterbi (work/EXPERIMENT-18.md).

States H, A, D with v8's costs -log(clip(q, 1e-6, 1)) / 25. Entering or leaving D costs P_D = 0.5 as in v8. A
switch between H and A arriving at frame t costs P * exp(-beta * c[t]). With P = 0.5 and beta = 0 this is exactly
`v2.methods.viterbi` (same tie rule: stay when not worse, else the lowest-index best origin).
"""
import math

import numpy as np

STATES = np.array(list('HAD'))
P_D = 0.5
GRID = [(p, b) for p in (.25, .5, 1.) for b in (0., 1., 2., 3., 4.)]
V8_SETTING = (.5, 0.)


def costs_of(q):
    return -np.log(np.clip(np.asarray(q, float), 1e-6, 1.)) / 25.


def decode(q, c, periods, penalty, beta):
    costs = costs_of(q)
    periods = np.asarray(periods)
    c = np.zeros(len(periods)) if c is None else np.asarray(c, float)
    if costs.shape != (len(periods), 3) or len(c) != len(periods) or not np.isfinite(costs).all():
        raise ValueError('q must be N by 3, finite, and match periods and c')
    pen = (penalty * np.exp(-beta * np.clip(c, 0., 1.))).tolist()
    out = np.full(len(periods), 'X', dtype='<U1')
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    pd = P_D
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] <= 0:
            continue
        cc = costs[start:end].tolist()
        pp = pen[start:end]
        n = end - start
        back = bytearray(3 * n)
        s0, s1, s2 = cc[0]
        for i in range(1, n):
            ph = pp[i]
            r0, r1, r2 = cc[i]
            k = 3 * i
            # H: from A (ph) or D (pd); lowest index wins ties
            a1, a2 = s1 + ph, s2 + pd
            o, oa = (a1, 1) if a1 <= a2 else (a2, 2)
            if s0 <= o:
                n0 = s0 + r0
            else:
                n0 = o + r0
                back[k] = oa
            # A: from H (ph) or D (pd)
            a0, a2b = s0 + ph, s2 + pd
            o, oa = (a0, 0) if a0 <= a2b else (a2b, 2)
            if s1 <= o:
                n1 = s1 + r1
                back[k + 1] = 1
            else:
                n1 = o + r1
                back[k + 1] = oa
            # D: from H or A, both pd
            o, oa = (s0 + pd, 0) if s0 <= s1 else (s1 + pd, 1)
            if s2 <= o:
                n2 = s2 + r2
                back[k + 2] = 2
            else:
                n2 = o + r2
                back[k + 2] = oa
            s0, s1, s2 = n0, n1, n2
        final = [s0, s1, s2]
        state = final.index(min(final))
        path = np.empty(n, np.int8)
        for i in range(n - 1, -1, -1):
            path[i] = state
            state = back[3 * i + state]
        out[start:end] = STATES[path]
    return out


assert math.isclose(V8_SETTING[0], P_D)
