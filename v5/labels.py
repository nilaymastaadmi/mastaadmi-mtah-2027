"""Label conventions for external data, from the competition's own possession definition.

CHALLENGE.md: a change of possession needs *controlled* possession; a simple touch, a deflection or
an aerial duel without subsequent control is not a change; contested balls stay with the last team.
'controlledK' absorbs an alive run shorter than K seconds when the surrounding alive runs (ignoring
nothing else) both belong to the other team: H H [A] H H -> H H H H H. Runs that touch a dead ball
or a half boundary are kept, because restarts give a clear possessor. 'raw' leaves labels unchanged.
"""
import numpy as np

FPS = 25


def runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(a, b, labels[a]) for a, b in zip(edges[:-1], edges[1:])]


def relabel(labels, periods, convention):
    if convention == 'raw':
        return labels
    if not convention.startswith('controlled'):
        raise ValueError(f'unknown convention {convention}')
    min_frames = int(round(float(convention[len('controlled'):]) * FPS))
    out = labels.copy()
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for s, e in zip(edges[:-1], edges[1:]):
        if periods[s] <= 0:
            continue
        seg = out[s:e]
        changed = True
        while changed:  # absorb shortest-first until stable
            changed = False
            rs = runs(seg)
            cands = [(b - a, i) for i, (a, b, lab) in enumerate(rs)
                     if lab in 'HA' and 0 < i < len(rs) - 1 and b - a < min_frames
                     and rs[i - 1][2] == rs[i + 1][2] and rs[i - 1][2] in 'HA' and rs[i - 1][2] != lab]
            if cands:
                _, i = min(cands)
                a, b, _ = rs[i]
                seg[a:b] = rs[i - 1][2]
                changed = True
        out[s:e] = seg
    return out
