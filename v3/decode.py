"""Three-state Viterbi with separate team-switch and live/dead-switch penalties (plain floats)."""
import numpy as np

STATES = np.array(list('HAD'))


def viterbi2(costs, periods, team_penalty, dead_penalty):
    costs, periods = np.asarray(costs, float), np.asarray(periods)
    if costs.ndim != 2 or costs.shape[1] != 3 or len(periods) != len(costs):
        raise ValueError('costs must be N by 3 and match periods')
    if not np.isfinite(costs).all() or min(team_penalty, dead_penalty) < 0:
        raise ValueError('invalid costs or penalties')
    t, d = float(team_penalty), float(dead_penalty)
    out = np.full(len(costs), 'X', dtype='<U1')
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] <= 0:
            continue
        c = costs[start:end].tolist()
        n = end - start
        back = bytearray(3 * n)
        h, a, x = c[0]
        for i in range(1, n):
            ch, ca, cd = c[i]
            k = 3 * i
            # to H: from H (0), A (t), D (d)
            nh, bh = h, 0
            if a + t < nh:
                nh, bh = a + t, 1
            if x + d < nh:
                nh, bh = x + d, 2
            na, ba = a, 1
            if h + t < na:
                na, ba = h + t, 0
            if x + d < na:
                na, ba = x + d, 2
            nd, bd = x, 2
            if h + d < nd:
                nd, bd = h + d, 0
            if a + d < nd:
                nd, bd = a + d, 1
            h, a, x = nh + ch, na + ca, nd + cd
            back[k], back[k + 1], back[k + 2] = bh, ba, bd
        final = [h, a, x]
        state = final.index(min(final))
        path = np.empty(n, np.int8)
        for i in range(n - 1, -1, -1):
            path[i] = state
            state = back[3 * i + state]
        out[start:end] = STATES[path]
    return out
