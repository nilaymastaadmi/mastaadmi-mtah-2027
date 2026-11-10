"""EXPERIMENT-18 ball-relative geometry per frame. No labels, identities or clock (work/EXPERIMENT-18.md).

Coordinates are normalised [0, 1] pitch units as everywhere in this repo; distances are in metres. A missing ball
gives NaN in every column except `ball_visible`. Windows never cross a half boundary.
"""
import warnings

import numpy as np

PITCH = np.array([105., 68.])
FPS = 25
NEAR_M = 5.
LAG = 3          # frames each side for speed and direction change
TEAM_LAG = 5     # frames back for "nearest team changed"

NAMES = ('ball_visible', 'ball_speed', 'ball_speed_change', 'ball_dir_cos',
         'home_d1', 'home_d2', 'away_d1', 'away_d2', 'd1_diff', 'home_rel_speed', 'away_rel_speed',
         'home_n5', 'away_n5', 'nearest_team', 'nearest_team_changed',
         'ball_touchline_m', 'ball_goalline_m', 'ball_off_pitch')


def _halves(periods):
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if periods[a] > 0]


def _velocity(xy, periods):
    """Centred 5-frame difference in m/s (as v6.control.velocity); NaN at half edges and where missing."""
    v = np.full(xy.shape, np.nan)
    for a, b in _halves(periods):
        seg = xy[a:b] * PITCH
        if b - a > 4:
            v[a + 2:b - 2] = (seg[4:] - seg[:-4]) * FPS / 4.
    return v


def _shift(x, k, periods):
    """x[t + k] inside the same half, NaN otherwise."""
    out = np.full(x.shape, np.nan)
    for a, b in _halves(periods):
        n = b - a
        if abs(k) >= n:
            continue
        if k >= 0:
            out[a:b - k] = x[a + k:b]
        else:
            out[a - k:b] = x[a:b + k]
    return out


def _two_smallest(d):
    s = np.sort(d, axis=1)
    first = s[:, 0] if s.shape[1] else np.full(len(d), np.inf)
    second = s[:, 1] if s.shape[1] > 1 else np.full(len(d), np.inf)
    return first, second


def geometry(players, teams, ball, periods):
    n = len(periods)
    periods = np.asarray(periods)
    visible = np.isfinite(ball).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        vb = _velocity(ball, periods)
        speed = np.linalg.norm(vb, axis=1)
        before, after = _shift(vb, -LAG, periods), _shift(vb, LAG, periods)
        sb, sa = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
        cos = np.sum(before * after, axis=1) / (sb * sa)
        cos = np.where((sb > 1.) & (sa > 1.), cos, np.nan)
        d = np.linalg.norm((players - ball[:, None, :]) * PITCH, axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        vp = _velocity(players, periods)
        cols = {}
        for t, name in ((0, 'home'), (1, 'away')):
            dt = d[:, teams == t]
            d1, d2 = _two_smallest(dt)
            k = np.argmin(dt, axis=1) if dt.shape[1] else np.zeros(n, int)
            rel = np.linalg.norm(vb - vp[:, teams == t][np.arange(n), k], axis=1) if dt.shape[1] else np.full(n, np.nan)
            cols[f'{name}_d1'], cols[f'{name}_d2'] = d1, d2
            cols[f'{name}_rel_speed'] = np.where(np.isfinite(d1), rel, np.nan)
            cols[f'{name}_n5'] = np.sum(dt <= NEAR_M, axis=1).astype(float)
        hd, ad = cols['home_d1'], cols['away_d1']
        nearest = np.where(np.isfinite(hd) | np.isfinite(ad), np.where(hd <= ad, 1., -1.), np.nan)
        prev = _shift(nearest, -TEAM_LAG, periods)
        changed = np.where(np.isfinite(nearest) & np.isfinite(prev), (nearest != prev).astype(float), np.nan)
        bm = ball * PITCH
        touch = np.minimum(bm[:, 1], PITCH[1] - bm[:, 1])
        goal = np.minimum(bm[:, 0], PITCH[0] - bm[:, 0])
        off = ((ball < 0) | (ball > 1)).any(axis=1).astype(float)
        out = {'ball_visible': visible.astype(float), 'ball_speed': speed, 'ball_speed_change': sa - sb,
               'ball_dir_cos': cos, 'home_d1': hd, 'home_d2': cols['home_d2'], 'away_d1': ad,
               'away_d2': cols['away_d2'], 'd1_diff': hd - ad, 'home_rel_speed': cols['home_rel_speed'],
               'away_rel_speed': cols['away_rel_speed'], 'home_n5': cols['home_n5'], 'away_n5': cols['away_n5'],
               'nearest_team': nearest, 'nearest_team_changed': changed,
               'ball_touchline_m': touch, 'ball_goalline_m': goal, 'ball_off_pitch': off}
    x = np.column_stack([out[k] for k in NAMES]).astype(float)
    x[~np.isfinite(x)] = np.nan
    x[~visible, 1:] = np.nan                     # no ball, no ball-relative geometry
    x[periods <= 0] = np.nan
    return x
