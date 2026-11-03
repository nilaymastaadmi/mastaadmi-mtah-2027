"""v3 features: v1's eight per-frame features plus temporal context. No labels, identities or clock.

All windows stay inside a half. Missing values are NaN (HistGradientBoosting handles them natively).
Offline decoder, so centred windows and look-ahead to the next visible ball are allowed.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learned_possession import inputs as v1_inputs  # noqa: E402  (frozen v1)

FPS = 25
PITCH = np.array([105., 68.])
WINDOWS = (12, 50, 125)  # half-widths in frames: about 0.5 s, 2 s, 5 s


def halves(periods):
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if periods[a] > 0]


def roll_mean(values, window, periods):
    out = np.full(len(values), np.nan)
    for a, b in halves(periods):
        v = values[a:b]
        ok = np.isfinite(v)
        s = np.r_[0., np.cumsum(np.where(ok, v, 0.))]
        c = np.r_[0, np.cumsum(ok)]
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - window), np.minimum(b - a, i + window + 1)
        n = c[hi] - c[lo]
        out[a:b] = np.where(n > 0, (s[hi] - s[lo]) / np.maximum(n, 1), np.nan)
    return out


def carry(values, periods, backward=False):
    """Last (or next) finite value within the half; NaN where none exists."""
    out = np.full(len(values), np.nan)
    for a, b in halves(periods):
        v = values[a:b][::-1] if backward else values[a:b]
        idx = np.where(np.isfinite(v), np.arange(len(v)), -1)
        idx = np.maximum.accumulate(idx)
        filled = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
        out[a:b] = filled[::-1] if backward else filled
    return out


def time_from_visible(visible, periods, backward=False, cap=60.):
    out = np.full(len(visible), cap)
    for a, b in halves(periods):
        v = visible[a:b][::-1] if backward else visible[a:b]
        i = np.arange(len(v))
        idx = np.maximum.accumulate(np.where(v, i, -1))
        t = np.where(idx >= 0, (i - idx) / FPS, cap)
        out[a:b] = np.minimum(t[::-1] if backward else t, cap)
    return out


def build(players, teams, ball, periods):
    x8, _ = v1_inputs(players, teams, ball, periods)
    visible = np.isfinite(ball).all(axis=1)
    diff = np.where(visible, x8[:, 0] - x8[:, 1], np.nan)  # home minus away nearest-distance cost
    metres = ball * PITCH
    edge = np.min(np.column_stack([metres[:, 0], PITCH[0] - metres[:, 0],
                                   metres[:, 1], PITCH[1] - metres[:, 1]]), axis=1)
    edge = np.where(visible, np.clip(edge, -10., 40.), np.nan)
    centre_x = np.where(visible, np.abs(metres[:, 0] - PITCH[0] / 2), np.nan)
    centre_y = np.where(visible, np.abs(metres[:, 1] - PITCH[1] / 2), np.nan)
    speed = np.where(visible, x8[:, 2], np.nan)
    d = np.linalg.norm((players - ball[:, None, :]) * PITCH, axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    near = [np.where(visible, np.sum(d[:, teams == t] <= 5., axis=1), np.nan) for t in (0, 1)]
    moving = np.linalg.norm(np.diff(players, axis=0) * PITCH, axis=2) * FPS
    moving = np.vstack([np.full((1, players.shape[1]), np.nan), np.minimum(moving, 12.)])
    moving[np.r_[False, periods[1:] != periods[:-1]]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-missing rows give NaN, as intended
        team_speed = [np.nanmean(moving[:, teams == t], axis=1) for t in (0, 1)]
    cols = [x8, diff[:, None], edge[:, None], centre_x[:, None], centre_y[:, None],
            near[0][:, None], near[1][:, None], team_speed[0][:, None], team_speed[1][:, None],
            time_from_visible(visible, periods)[:, None],
            time_from_visible(visible, periods, backward=True)[:, None],
            carry(diff, periods)[:, None], carry(diff, periods, backward=True)[:, None],
            carry(edge, periods)[:, None], carry(edge, periods, backward=True)[:, None]]
    for w in WINDOWS:
        for v in (diff, speed, x8[:, 7], visible.astype(float), x8[:, 3]):
            cols.append(roll_mean(v, w, periods)[:, None])
    return np.hstack(cols).astype(np.float64)
