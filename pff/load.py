"""PFF FC World Cup 2022 loader (EXPERIMENT-19): 25 fps arrays shaped like `v3.skillcorner.load`, plus H/A/D truth.

    .venv/bin/python -m pff.load                 # caches every match under ignored external/pff-cache/load/

Data: ignored external/pff/ (tracking/*.jsonl.bz2, metadata/*.json). Nothing here is published.

Rules (work/EXPERIMENT-19.md, fixed before any score):
- Periods 1 and 2 only (extra time and shoot-outs excluded). One record per frameNum (the first; PFF duplicates a
  frame when it carries several possession events, positions are identical).
- Players: the raw `homePlayers` / `awayPlayers` (not the Kalman-smoothed copies), one slot per (team, jersey); if a
  jersey appears twice in one frame the higher confidence wins, then the first. Teams: home 0, away 1.
- Ball: the raw `balls` entry when exactly one is given (VISIBLE or ESTIMATED); none or several gives NaN. The
  event-smoothed `ballsSmoothed` is never read: PFF builds it from the event data, so it would leak the truth.
- Coordinates: metres centred on the pitch -> (x + L/2) / L, (y + W/2) / W with L, W from the metadata pitch, the
  same mapping as the SkillCorner loader.
- Resample about 29.97 -> 25 fps per period: output time t0 + k / 25 from the first frame of the period, positions
  linearly interpolated between the two neighbouring native frames when both are finite and at most 2 frameNums
  apart (else NaN); labels from the nearest native frame. The two halves are joined by one ungraded separator frame
  (period 0, label X), as for SkillCorner.
- Truth: `pff.truth.native_labels` from the game events embedded in the tracking frames.
"""
import bz2
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pff.truth import native_labels  # noqa: E402

PFF = ROOT / 'external/pff'
CACHE = ROOT / 'external/pff-cache/load'
FPS = 25
MAX_GAP = 2
CONF = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}


def match_ids():
    return sorted(int(p.name.split('.')[0]) for p in (PFF / 'tracking').glob('*.jsonl.bz2'))


def metadata(match):
    m = json.loads((PFF / 'metadata' / f'{match}.json').read_text())
    m = m[0] if isinstance(m, list) else m
    date = m['date'][:10]
    pitches = [p for p in m['stadium']['pitches']
               if p['startDate'] <= date and (p['endDate'] is None or date <= p['endDate'])] or m['stadium']['pitches']
    p = pitches[-1]
    return {'pitch': np.array([float(p['length']), float(p['width'])]), 'fps': float(m['fps']),
            'home': m['homeTeam']['name'], 'away': m['awayTeam']['name']}


def read_frames(match):
    """Native records, periods 1 and 2, first record per frameNum, sorted; plus the dropped duplicates (their game
    events still count, review fix 11 Nov)."""
    out, seen, extra = [], set(), []
    with bz2.open(PFF / 'tracking' / f'{match}.jsonl.bz2', 'rt') as handle:
        for line in handle:
            r = json.loads(line)
            if r.get('period') not in (1, 2):
                continue
            f = int(r['frameNum'])
            if f in seen:
                extra.append(r)
                continue
            seen.add(f)
            out.append(r)
    out.sort(key=lambda r: r['frameNum'])
    return out, extra


def _jersey(p):
    j = p.get('jerseyNum')
    return None if j is None or p.get('x') is None or p.get('y') is None else str(j)


def native_arrays(records):
    slots = {}
    for side, team in (('homePlayers', 0), ('awayPlayers', 1)):
        js = {_jersey(p) for r in records for p in (r.get(side) or [])} - {None}
        for j in sorted(js, key=lambda s: (len(s), s)):
            slots[(team, j)] = len(slots)
    n = len(records)
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    stats = {'ball_visible': 0, 'ball_estimated': 0, 'ball_none': 0, 'ball_several': 0, 'jersey_duplicates': 0}
    for i, r in enumerate(records):
        for side, team in (('homePlayers', 0), ('awayPlayers', 1)):
            best = {}
            for p in r.get(side) or []:
                j = _jersey(p)
                if j is None:
                    continue
                rank = CONF.get(p.get('confidence'), 0)
                if j in best:
                    stats['jersey_duplicates'] += 1
                    if rank <= best[j][0]:
                        continue
                best[j] = (rank, p['x'], p['y'])
            for j, (_, x, y) in best.items():
                players[i, slots[(team, j)]] = (x, y)
        b = r.get('balls') or []
        if len(b) == 1 and b[0].get('x') is not None and b[0].get('y') is not None:
            ball[i] = (b[0]['x'], b[0]['y'])
            stats['ball_visible' if b[0].get('visibility') == 'VISIBLE' else 'ball_estimated'] += 1
        else:
            stats['ball_none' if not b else 'ball_several'] += 1
    teams = np.array([t for t, _ in sorted(slots, key=slots.get)])
    return players, ball, teams, stats


def resample(frames, values, fps, kind):
    """frames: native frameNums of one period (sorted). Returns 25 fps values from the first frame's time."""
    t = (frames - frames[0]) / fps
    n_out = int(np.floor(t[-1] * FPS)) + 1
    tt = np.arange(n_out) / FPS
    hi = np.clip(np.searchsorted(t, tt, side='left'), 0, len(t) - 1)
    lo = np.clip(hi - 1, 0, len(t) - 1)
    exact = np.abs(t[hi] - tt) <= 1e-9          # absolute only (review fix 11 Nov: isclose's rtol grew with time)
    lo = np.where(exact, hi, lo)
    if kind == 'nearest':
        pick = np.where(np.abs(t[hi] - tt) <= np.abs(tt - t[lo]), hi, lo)
        return values[pick]
    span = t[hi] - t[lo]
    frac = np.where(span > 0, (tt - t[lo]) / np.where(span > 0, span, 1.), 0.)
    frac = frac.reshape((-1,) + (1,) * (values.ndim - 1))
    out = values[lo] + (values[hi] - values[lo]) * frac
    far = (frames[hi] - frames[lo]) > MAX_GAP
    out[far] = np.nan
    return out


def load(match, use_cache=True):
    path = CACHE / f'{match}.npz'
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d['players'], d['ball'] = d['players'].astype(np.float32).astype(float), d['ball'].astype(np.float32).astype(float)
        d['stats'] = json.loads(str(d['stats']))
        return d
    meta = metadata(match)
    records, extra = read_frames(match)
    frames = np.array([r['frameNum'] for r in records])
    periods = np.array([r['period'] for r in records])
    players, ball, teams, stats = native_arrays(records)
    labels, lstats = native_labels(records, extra)
    pitch = meta['pitch']
    players = (players + pitch / 2) / pitch
    ball = (ball + pitch / 2) / pitch
    blocks = []
    for half in (1, 2):
        sel = periods == half
        f = frames[sel]
        blocks.append({'players': resample(f, players[sel], meta['fps'], 'linear'),
                       'ball': resample(f, ball[sel], meta['fps'], 'linear'),
                       'labels': resample(f, labels[sel], meta['fps'], 'nearest')})
        blocks[-1]['periods'] = np.full(len(blocks[-1]['labels']), half)
    sep = {'players': np.full((1,) + players.shape[1:], np.nan), 'ball': np.full((1, 2), np.nan),
           'labels': np.array(['X']), 'periods': np.zeros(1, int)}
    out = {k: np.concatenate([blocks[0][k], sep[k], blocks[1][k]]) for k in sep}
    out['players'] = out['players'].astype(np.float32).astype(float)   # same values as a cached load (review fix)
    out['ball'] = out['ball'].astype(np.float32).astype(float)
    out['teams'], out['pitch'] = teams, pitch
    out['stats'] = {**stats, **lstats, 'native_frames': int(len(records)), 'duplicate_records': len(extra),
                    'fps': meta['fps'], 'frame_gaps_over_2': int(sum(int(np.sum(np.diff(frames[periods == h]) > MAX_GAP))
                                                                     for h in (1, 2)))}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, players=out['players'].astype(np.float32), ball=out['ball'].astype(np.float32),
                        labels=out['labels'], periods=out['periods'], teams=out['teams'], pitch=out['pitch'],
                        stats=np.array(json.dumps(out['stats'])))
    tmp.rename(path)
    return out


def _cache(match):
    d = load(match)
    return match, d['stats']


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    todo = [m for m in match_ids() if not (CACHE / f'{m}.npz').exists()]
    with ProcessPoolExecutor(8) as pool:
        for m, s in pool.map(_cache, todo):
            print('cached', m, flush=True)
    print(len(match_ids()), 'matches ready')
