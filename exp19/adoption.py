"""EXPERIMENT-19 amendment 2, label-free adoption check on the competition match. LOCAL MACHINE ONLY.

    OMP_NUM_THREADS=1 .venv/bin/python -m exp19.adoption --data-dir <data> --v8 Mastaadmi.csv \
        --v11 work/v11-evaluation/v11_<stamp>_run1.csv          # writes work/exp19/adoption-match.json

Reads only the tactical and broadcast tracking (through v8's fused-ball path, as v10/v11 inputs) and the two
prediction CSVs. Never reads ground truth, tags or evaluation files. Prints aggregates only.
Definitions (work/EXPERIMENT-19.md, amendment 2): possession start = graded frame predicted H or A whose label differs
from the previous frame in the same half; quiet stretch = maximal run in one half with no fused ball for at least
125 frames and mean player speed below 1.5 m/s over the run. Fail if v11 has >= 10% more possessions per minute than
v8 and at least half of its extra starts fall inside quiet stretches (amendment 2), or if its starts inside quiet
stretches number at least 1.5 times v8's (amendment 3). --v10 adds v10's numbers for reference (no rule uses them).
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp18.geometry import _velocity  # noqa: E402

FPS = 25
MIN_RUN, MAX_SPEED = 125, 1.5


def read_labels(path, n):
    rows = Path(path).read_text().splitlines()[1:]
    labels = np.array([r.split(',')[1] for r in rows])
    if len(labels) != n:
        raise SystemExit(f'{path}: {len(labels)} rows, expected {n}')
    return labels


def quiet_mask(ball_missing, speed, periods):
    out = np.zeros(len(periods), bool)
    edges = np.r_[0, np.flatnonzero((ball_missing[1:] != ball_missing[:-1]) | (periods[1:] != periods[:-1])) + 1,
                  len(periods)]
    for a, b in zip(edges[:-1], edges[1:]):
        if periods[a] > 0 and ball_missing[a] and b - a >= MIN_RUN:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                s = np.nanmean(speed[a:b])
            if np.isfinite(s) and s < MAX_SPEED:
                out[a:b] = True
    return out


def starts(labels, periods):
    s = np.zeros(len(labels), bool)
    s[1:] = (periods[1:] > 0) & (periods[1:] == periods[:-1]) & np.isin(labels[1:], ['H', 'A']) & (labels[1:] != labels[:-1])
    return s


def stats(labels, periods, quiet):
    graded = periods > 0
    st = starts(labels, periods)
    minutes = graded.sum() / FPS / 60
    return {'possessions_per_minute': float(st.sum() / minutes), 'starts': int(st.sum()),
            'starts_in_quiet': int((st & quiet).sum()), 'share_in_quiet': float((st & quiet).sum() / max(1, st.sum())),
            'd_share': float(np.mean(labels[graded] == 'D'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--v8', type=Path, required=True)
    p.add_argument('--v11', type=Path, required=True)
    p.add_argument('--v10', type=Path, help='optional reference CSV (reported only)')
    args = p.parse_args()
    from possession import load_atd
    from v8 import align as al
    from v8.broadcast_audit import load_broadcast
    players, teams, ball, periods = load_atd(args.data_dir)
    bp, bb = load_broadcast(args.data_dir)
    fused, _ = al.fuse_ball(ball, bb, al.align(players, bp, periods))
    periods = np.asarray(periods)
    missing = ~np.isfinite(fused).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        speed = np.nanmean(np.linalg.norm(_velocity(players, periods), axis=2), axis=1)
    quiet = quiet_mask(missing, speed, periods)
    res = {'quiet_share_of_graded': float(quiet[periods > 0].mean()),
           'quiet_stretches': int(np.sum(np.diff(np.r_[0, quiet.astype(int)]) == 1))}
    for name, path in (('v8', args.v8), ('v11', args.v11)) + ((('v10', args.v10),) if args.v10 else ()):
        res[name] = {**stats(read_labels(path, len(periods)), periods, quiet),
                     'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    a, b = res['v8'], res['v11']
    extra = b['starts'] - a['starts']
    extra_quiet = b['starts_in_quiet'] - a['starts_in_quiet']
    more = b['possessions_per_minute'] >= 1.1 * a['possessions_per_minute']
    concentrated = extra > 0 and extra_quiet >= .5 * extra
    res['extra_starts'], res['extra_starts_in_quiet'] = extra, extra_quiet
    res['fail_amendment2'] = bool(more and concentrated)
    res['fail_amendment3'] = bool(b['starts_in_quiet'] >= 1.5 * a['starts_in_quiet'])
    res['fail'] = res['fail_amendment2'] or res['fail_amendment3']
    (ROOT / 'work/exp19/adoption-match.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
