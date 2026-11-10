"""EXPERIMENT-18 arm B: possession-change detector (HistGradientBoosting on v6 columns + geometry + wide differences).

    OMP_NUM_THREADS=4 .venv/bin/python -m exp18.arm_b [--fold N | --final] [--out DIR]
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.experiment import HGB  # noqa: E402
from exp18 import views as vw  # noqa: E402

OUT = ROOT / 'external/exp18/arm_b'
SUMMARY = ROOT / 'work/exp18/arm_b_summary.json'
DIFF_COLS = np.r_[np.arange(vw.CONTROL.start, vw.CONTROL.stop), np.arange(vw.N_V6, 90)]
DIFFS = (12, 25, 50)


def features(x, periods):
    n = len(x)
    periods = np.asarray(periods)
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, n]
    lens = np.diff(edges)
    start, end = np.repeat(edges[:-1], lens), np.repeat(edges[1:], lens)
    i = np.arange(n)
    out = [x[:, :90]]
    for d in DIFFS:
        ok = (i - d >= start) & (i + d < end)
        f = np.full((n, len(DIFF_COLS)), np.nan, np.float32)
        f[ok] = x[np.minimum(i + d, n - 1)[ok]][:, DIFF_COLS] - x[np.maximum(i - d, 0)[ok]][:, DIFF_COLS]
        out.append(f)
    return np.hstack(out).astype(np.float32)


def target(labels, periods):
    return vw.near(vw.change_events(labels), 12, periods)


def row_mask(labels, periods, visible, is_sc):
    m = (np.arange(len(labels)) % 5 == 0) & (np.asarray(labels) != 'X') & (np.asarray(periods) > 0)
    return m & np.asarray(visible, bool) if is_sc else m


def train_xy(views):
    xs, ys = [], []
    for v in views:
        d = vw.load(v)
        m = row_mask(d['labels'], d['periods'], d['visible'], v[0] == 'sc')
        xs.append(features(d['x'], d['periods'])[m])
        ys.append(target(d['labels'], d['periods'])[m])
    return np.concatenate(xs), np.concatenate(ys)


def fit(views):
    x, y = train_xy(views)
    return HistGradientBoostingClassifier(**HGB).fit(x, y), len(y)


def run_fold(f, out):
    done = out / f'fold{f}.done'
    if done.exists():
        return
    t0 = time.time()
    (out / 'models').mkdir(parents=True, exist_ok=True)
    model, rows = fit(vw.train_views(f))
    for v in vw.eval_views(f):
        d = vw.load(v)
        c = model.predict_proba(features(d['x'], d['periods']))[:, 1].astype(np.float32)
        c[np.asarray(d['periods']) <= 0] = 0
        tmp = out / f'{vw.name(v)}.tmp.npz'
        np.savez(tmp, c=c)
        tmp.rename(out / f'{vw.name(v)}.npz')
    with open(out / 'models' / f'fold{f}.pkl', 'wb') as fh:
        pickle.dump(model, fh)
    done.write_text(json.dumps({'train_rows': int(rows), 'wall_s': round(time.time() - t0, 1)}))


def run_final(out):
    (out / 'models').mkdir(parents=True, exist_ok=True)
    model, rows = fit(vw.all_train_views())
    p = out / 'models' / 'final.pkl'
    p.write_bytes(pickle.dumps(model))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'models' / 'final.sha256').write_text(h + '\n')
    print('final train_rows', rows, 'sha256', h)


def summarize(out):
    per = {'real': [], 'clean': [], 'gaps': []}
    digests = {}
    for v in vw.eval_views():
        z = out / f'{vw.name(v)}.npz'
        digests[vw.name(v)] = hashlib.sha256(z.read_bytes()).hexdigest()
        d = vw.load(v)
        with np.load(z) as zz:
            c = zz['c']
        m = (d['labels'] != 'X') & (d['periods'] > 0)
        y = target(d['labels'], d['periods'])[m]
        per[vw.setting_of(v)].append((roc_auc_score(y, c[m]), average_precision_score(y, c[m]), y.mean()))
    folds = {f: json.loads((out / f'fold{f}.done').read_text()) for f in range(5)}
    s = {'settings': {k: {'views': len(r), 'mean_auc': float(np.mean([a[0] for a in r])),
                          'mean_ap': float(np.mean([a[1] for a in r])), 'positive_share': float(np.mean([a[2] for a in r]))}
                      for k, r in per.items()},
         'train_rows': {f: v['train_rows'] for f, v in folds.items()},
         'wall_s': {f: v['wall_s'] for f, v in folds.items()}, 'c_sha256': digests}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != 'c_sha256'}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if a.final:
        run_final(a.out)
    else:
        folds = range(5) if a.fold is None else [a.fold]
        for f in folds:
            run_fold(f, a.out)
        if all((a.out / f'fold{f}.done').exists() for f in range(5)):
            summarize(a.out)
