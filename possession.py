"""Local possession inference. No labels are read by this module."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'starter-repo'))
from starter.baseline import smooth, write_submission


def features(players, teams, ball, periods, fps=25, pitch=(105., 68.), missing_dead_cost=.5):
    players, ball = np.asarray(players, float), np.asarray(ball, float)
    teams, periods = np.asarray(teams), np.asarray(periods)
    n = len(ball)
    if (players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2
            or ball.shape != (n, 2) or len(teams) != players.shape[1]
            or periods.shape != (n,) or not np.isin(teams, [0, 1]).all()
            or not (np.any(teams == 0) and np.any(teams == 1))):
        raise ValueError('inconsistent tracking arrays or team assignments')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('invalid frame rate or pitch')
    visible = np.isfinite(ball).all(axis=1)
    distances = np.linalg.norm((players - ball[:, None, :]) * pitch, axis=2)
    distances[~np.isfinite(distances)] = np.inf
    nearest = np.column_stack([np.min(distances[:, teams == team], axis=1)
                               for team in [0, 1]])
    owner = np.argmin(nearest, axis=1).astype(np.int8)
    owner[~visible | ~np.isfinite(nearest).any(axis=1)] = -1
    speed = np.full(n, np.nan)
    valid_step = visible[1:] & visible[:-1] & (periods[1:] == periods[:-1])
    speed[1:][valid_step] = np.linalg.norm(np.diff(ball, axis=0)[valid_step] * pitch,
                                         axis=1) * fps
    stationary = np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isfinite(speed[i]) and speed[i] < .4:
            stationary[i] = stationary[i - 1] + 1
    outside = visible & ((ball < 0).any(axis=1) | (ball > 1).any(axis=1))
    costs = np.minimum(nearest / 2., 6.)
    dead = np.full(n, 4.)
    dead[stationary >= fps] = .25
    dead[outside] = 0.
    costs[outside] = 4.
    costs[~visible] = .5
    dead[~visible] = missing_dead_cost
    # No tracked players is missing evidence, not evidence of dead ball.
    costs[visible & ~np.isfinite(nearest).any(axis=1)] = .5
    return np.column_stack([costs, dead]) / fps, owner


def decode(costs, periods, penalty=15.):
    costs, periods = np.asarray(costs, float), np.asarray(periods)
    if costs.ndim != 2 or costs.shape[1] != 3 or not len(costs):
        raise ValueError('costs must be a nonempty N by 3 array')
    if (len(periods) != len(costs) or not np.isfinite(costs).all()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('invalid costs, periods or penalty')
    out = np.full(len(costs), 'X', dtype='<U1')
    starts = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    transition = (1 - np.eye(3)) * penalty
    for start, end in zip(starts[:-1], starts[1:]):
        if periods[start] <= 0:
            continue
        back = np.empty((end - start, 3), dtype=np.uint8)
        score = costs[start].copy()
        for i in range(start + 1, end):
            options = score[:, None] + transition
            prev = np.argmin(options, axis=0)
            back[i - start] = prev
            score = options[prev, np.arange(3)] + costs[i]
        state = int(np.argmin(score))
        for i in range(end - 1, start - 1, -1):
            out[i] = 'HAD'[state]
            if i > start:
                state = int(back[i - start, state])
    return out


def baseline(owner, periods, window=50):
    out = np.full(len(owner), 'X', dtype='<U1')
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] > 0:
            out[start:end] = smooth(owner[start:end], window)
    return out


def load_atd(data_dir):
    from games import asset_paths, source_frames, half_video_blocks
    paths = asset_paths(data_dir=str(data_dir))
    root = ET.parse(paths['atd_meta']).getroot()
    ids = [t.get('id') for t in root.iter('Team')]
    slots = [p.get('teamId') for p in root.iter('Player')]
    if len(ids) != 2 or any(team not in ids for team in slots):
        raise ValueError('ATD must assign every slot to one of two teams')
    teams = np.array([ids.index(team) for team in slots])
    n = source_frames()
    players = np.full((n, len(teams), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    seen = np.zeros(n, bool)
    with open(paths['atd_raw']) as handle:
        for line in handle:
            fields = line.strip().split(':')
            if len(fields) != 3:
                raise ValueError('malformed ATD line')
            i = int(fields[0]) - 1
            if i < 0 or i >= n or seen[i]:
                raise ValueError('ATD frame out of bounds or duplicated')
            groups = fields[1].split(';')
            if len(groups) != len(teams):
                raise ValueError('ATD track-slot count differs from metadata')
            players[i] = [[float(v) for v in g.split(',')[:2]] for g in groups]
            ball[i] = [float(v) for v in fields[2].split(',')[:2]]
            seen[i] = True
    if not seen.all():
        raise ValueError('ATD does not cover all tactical frames')
    periods = np.zeros(n, int)
    for half, start, end in half_video_blocks('BAR-ATH'):
        periods[start:end + 1] = half
    return players, teams, ball, periods


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--penalty', type=float, required=True)
    args = parser.parse_args()
    arrays = load_atd(args.data_dir)
    costs, _ = features(*arrays, pitch=(105.3, 68.))
    labels = decode(costs, arrays[-1], args.penalty)
    write_submission(str(args.out), labels)
    print('Wrote full-length candidate; competition labels were not read.')
