"""Deterministic tracking degradation, fixed in EXPERIMENT-04 before any outcome.

Calibrated only to aggregate, unlabelled input diagnostics already recorded in
work/first-assessment/domain-audit.json (competition missing-ball share 40.7%, missing-run median
10.72 s, p90 33.8 s, max 221.5 s). No competition labels, frames or per-frame values are used.
"""
import hashlib

import numpy as np
from scipy.signal import lfilter

SEED = 20261102
FPS = 25
TARGET_MISSING = .41
RUN_MEDIAN_S, RUN_P90_S, RUN_MAX_S = 10.72, 33.8, 221.5
RUN_SIGMA = float(np.log(RUN_P90_S / RUN_MEDIAN_S) / 1.2815515655446004)
PLAYER_MISSING, PLAYER_MEDIAN_S, PLAYER_SIGMA, PLAYER_MAX_S = .08, 2., 1., 60.
PLAYER_JITTER_M, BALL_JITTER_M, JITTER_PHI = .3, .5, .96
OUTLIER_SHARE, OUTLIER_RUN = .02, 25
CONDITIONS = ('natural', 'gaps', 'contested')


def seed_for(match, condition, purpose):
    text = f'{match}|{condition}|{purpose}|{SEED}'.encode()
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def run_length(rng, median_s, sigma, max_s):
    seconds = min(max(rng.lognormal(np.log(median_s), sigma), .2), max_s)
    return max(1, int(round(seconds * FPS)))


def place_runs(eligible, weights, target, rng, median_s, sigma, max_s):
    """Missing runs placed with start probability proportional to weights until target share."""
    eligible = np.asarray(eligible, bool)
    missing = np.zeros(len(eligible), bool)
    total = eligible.sum()
    if total == 0:
        return missing
    w = np.where(eligible, np.asarray(weights, float), 0.)
    while missing[eligible].sum() < target * total:
        p = np.where(missing, 0., w)
        if p.sum() <= 0:
            break
        start = rng.choice(len(p), p=p / p.sum())
        missing[start:start + run_length(rng, median_s, sigma, max_s)] = True
    return missing & eligible


def contested_weights(players, ball, z, teams, pitch):
    distances = np.linalg.norm((players - ball[:, None, :]) * pitch, axis=2)
    crowded = np.sum(np.nan_to_num(distances, nan=np.inf) <= 5., axis=1) >= 3
    aerial = np.nan_to_num(z, nan=0.) >= 1.5
    return 1. + 4. * crowded + 4. * aerial


def ar1(rng, shape, sigma):
    innovations = rng.normal(0., sigma * np.sqrt(1 - JITTER_PHI ** 2), shape)
    return lfilter([1.], [1., -JITTER_PHI], innovations, axis=0)


def degrade(arrays, match, condition, purpose):
    """Return (players, ball) observed under the named condition. Labels are not touched."""
    players, ball = arrays['players'].copy(), arrays['ball'].copy()
    if condition == 'natural':
        return players, ball
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition}')
    rng = np.random.default_rng(seed_for(match, condition, purpose))
    pitch = np.asarray(arrays['pitch'], float)
    periods, z, teams = arrays['periods'], arrays['z'], arrays['teams']
    weights = (contested_weights(arrays['players'], arrays['ball'], z, teams, pitch)
               if condition == 'contested' else np.ones(len(ball)))
    for half in (1, 2):
        rows = np.flatnonzero(periods == half)
        p, b = players[rows], ball[rows]
        p += ar1(rng, p.shape, PLAYER_JITTER_M) / pitch
        b += ar1(rng, b.shape, BALL_JITTER_M) / pitch
        visible = np.isfinite(b).all(axis=1)
        outliers = place_runs(visible, np.ones(len(b)), OUTLIER_SHARE, rng,
                              OUTLIER_RUN / FPS, 1e-9, OUTLIER_RUN / FPS)
        for start in np.flatnonzero(outliers & ~np.r_[False, outliers[:-1]]):
            end = start + np.argmax(~outliers[start:]) if (~outliers[start:]).any() else len(b)
            b[start:end] = rng.uniform(0., 1., 2)
        gone = place_runs(np.isfinite(b).all(axis=1), weights[rows], TARGET_MISSING, rng,
                          RUN_MEDIAN_S, RUN_SIGMA, RUN_MAX_S)
        b[gone] = np.nan
        for slot in range(p.shape[1]):
            on = np.isfinite(p[:, slot]).all(axis=1)
            drop = place_runs(on, np.ones(len(p)), PLAYER_MISSING, rng,
                              PLAYER_MEDIAN_S, PLAYER_SIGMA, PLAYER_MAX_S)
            p[drop, slot] = np.nan
        players[rows], ball[rows] = p, b
    return players, ball
