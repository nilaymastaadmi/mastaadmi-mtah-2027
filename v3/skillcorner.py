"""Load SkillCorner open data (MIT, (c) SkillCorner) into v1-compatible 25 fps arrays plus labels.

Rules fixed in EXPERIMENT-05 from pilot match 1925299 only:
- periods from match.json match_periods; each period upsampled 10 -> 25 fps by linear interpolation
  between neighbouring 10 fps frames when both are finite (else NaN);
- players: SkillCorner extrapolated positions, slots by player_id, teams from match.json;
- ball: only where is_detected is true;
- labels: nearest 10 fps frame possession.group (home H, away A, none D); optional phases-of-play
  labels for sensitivity. Labels never enter features.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'external/skillcorner/data/matches'
TRACKING = ROOT / 'external/skillcorner-lfs'
PILOT = 1925299
SRC_FPS, FPS = 10, 25


def match_ids():
    ids = sorted(m['id'] for m in json.loads((ROOT / 'external/skillcorner/data/matches.json').read_text()))
    return [i for i in ids if i != PILOT]


def parse(match, meta=META, tracking=TRACKING):
    info = json.loads((meta / str(match) / f'{match}_match.json').read_text())
    home = info['home_team']['id']
    team_of = {p['id']: (0 if p['team_id'] == home else 1) for p in info['players']}
    slots = {pid: k for k, pid in enumerate(sorted(team_of, key=lambda p: (team_of[p], p)))}
    frames = {}
    with (tracking / f'{match}_tracking_extrapolated.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            frames[r['frame']] = r
    return info, team_of, slots, frames


def source_arrays(frames, slots, start, end):
    n = end - start + 1
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    group = np.full(n, 'D', dtype='<U1')
    for i, f in enumerate(range(start, end + 1)):
        r = frames.get(f)
        if r is None:
            continue
        for p in r.get('player_data') or []:
            k = slots.get(p['player_id'])
            if k is not None and p['x'] is not None and p['y'] is not None:
                players[i, k] = (p['x'], p['y'])
        b = r.get('ball_data') or {}
        if b.get('is_detected') and b.get('x') is not None and b.get('y') is not None:
            ball[i] = (b['x'], b['y'])
        group[i] = {'home team': 'H', 'away team': 'A'}.get((r.get('possession') or {}).get('group'), 'D')
    return players, ball, group


def upsample(values, n_out):
    """Linear interpolation from 10 fps positions to 25 fps; NaN unless both neighbours are finite."""
    pos = np.arange(n_out) * SRC_FPS / FPS
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(values) - 1)
    frac = (pos - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    a, b = values[lo], values[hi]
    out = a + (b - a) * frac
    exact = (frac == 0).reshape(-1)
    out[exact] = a[exact]
    return out


def nearest(values, n_out):
    idx = np.minimum(np.rint(np.arange(n_out) * SRC_FPS / FPS).astype(int), len(values) - 1)
    return values[idx]


def phase_labels(match, start, end, home, meta=META):
    phases = pd.read_csv(meta / str(match) / f'{match}_phases_of_play.csv')
    out = np.full(end - start + 1, 'D', dtype='<U1')
    for _, p in phases.iterrows():
        a, b = max(int(p.frame_start), start), min(int(p.frame_end), end)
        if a <= b:
            out[a - start:b - start + 1] = 'H' if p.team_in_possession_id == home else 'A'
    return out


def load(match, meta=META, tracking=TRACKING, with_phases=False):
    info, team_of, slots, frames = parse(match, meta, tracking)
    pitch = np.array([float(info['pitch_length']), float(info['pitch_width'])])
    blocks = []
    for half, period in enumerate(sorted(info['match_periods'], key=lambda p: p['period'])[:2], start=1):
        start, end = int(period['start_frame']), int(period['end_frame'])
        players, ball, group = source_arrays(frames, slots, start, end)
        n_out = int(np.floor((end - start) * FPS / SRC_FPS)) + 1
        block = {'players': (upsample(players, n_out) + pitch / 2) / pitch,
                 'ball': (upsample(ball, n_out) + pitch / 2) / pitch,
                 'labels': nearest(group, n_out), 'periods': np.full(n_out, half)}
        if with_phases:
            block['phase_labels'] = nearest(phase_labels(match, start, end, info['home_team']['id'], meta), n_out)
        blocks.append(block)
    sep = {k: (np.full((1,) + v.shape[1:], np.nan) if v.dtype.kind == 'f' else
               (np.zeros(1, int) if k == 'periods' else np.array(['X'])))
           for k, v in blocks[0].items()}
    parts = [blocks[0], sep, blocks[1]]
    out = {k: np.concatenate([p[k] for p in parts]) for k in blocks[0]}
    out['teams'] = np.array([team_of[p] for p in sorted(slots, key=slots.get)])
    out['pitch'] = pitch
    return out
