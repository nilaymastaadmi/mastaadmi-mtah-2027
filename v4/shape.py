"""Team-shape features: available even when the ball is missing. No labels, identities or clock.

Attack direction per half is inferred from each team's median centroid x over that half (the team
whose players sit at lower x defends the low-x goal). Coordinates are converted to metres in each
team's attacking frame, so features mean the same thing in both halves.
"""
import warnings

import numpy as np

PITCH = np.array([105., 68.])
FPS = 25


def _halves(periods):
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if periods[a] > 0]


def team_shape(players, teams, periods):
    n = len(periods)
    feats = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for t, name in [(0, 'home'), (1, 'away')]:
            xy = players[:, teams == t] * PITCH
            att = np.full(n, np.nan)  # +1 attacks towards high x
            for a, b in _halves(periods):
                mine = np.nanmedian(np.nanmean(xy[a:b, :, 0], axis=1))
                other = np.nanmedian(np.nanmean((players[a:b, teams != t] * PITCH)[..., 0], axis=1))
                att[a:b] = 1. if mine < other else -1.
            x = np.where(att[:, None] > 0, xy[..., 0], PITCH[0] - xy[..., 0])  # metres from own goal line
            y = xy[..., 1]
            feats[f'{name}_cx'] = np.nanmean(x, axis=1)
            feats[f'{name}_sx'] = np.nanstd(x, axis=1)
            feats[f'{name}_sy'] = np.nanstd(y, axis=1)
            feats[f'{name}_len'] = np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)
            feats[f'{name}_wid'] = np.nanpercentile(y, 90, axis=1) - np.nanpercentile(y, 10, axis=1)
            feats[f'{name}_back'] = np.nanpercentile(x, 20, axis=1)   # defensive line height proxy
            feats[f'{name}_n'] = np.sum(np.isfinite(x), axis=1).astype(float)
            v = np.full(n, np.nan)
            v[FPS:] = (feats[f'{name}_cx'][FPS:] - feats[f'{name}_cx'][:-FPS])  # advance per second
            v[np.r_[np.zeros(FPS, bool), periods[FPS:] != periods[:-FPS]]] = np.nan
            feats[f'{name}_adv'] = v
    for k in ['cx', 'sx', 'sy', 'len', 'wid', 'back', 'adv']:
        feats[f'diff_{k}'] = feats[f'home_{k}'] - feats[f'away_{k}']
    names = sorted(feats)
    return np.column_stack([feats[k] for k in names]), names
