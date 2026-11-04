"""Out-of-fold probabilities for the v4 recipe (variant C), stored once for fast decoder/label studies.

    OMP_NUM_THREADS=4 .venv/bin/python -m v5.oof [--labels raw|controlled2]

Same folds, data and training rule as EXPERIMENT-06. Writes ignored caches under
external/oof/<labels>/<dataset>-<match>-<condition>.npz with q (N x 3, HAD), owner, periods, labels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, methods as mt  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v5.labels import relabel  # noqa: E402


def train(fold, convention):
    xs, ys = [], []
    for m in dv.DFL:
        if dv.fold_of(dv.DFL, m) == fold:
            continue
        for c in dg.CONDITIONS:
            d = dv.dfl_data(m, c, 'train')
            y = relabel(d['labels'], d['periods'], convention)
            g = y != 'X'
            xs.append(dv.columns(d['x'], d['shape'], 'C')[g])
            ys.append(y[g])
    for m in dv.SC:
        if dv.fold_of(dv.SC, m) == fold:
            continue
        d = dv.sc_data(m)
        y = relabel(d['labels'], d['periods'], convention)
        g = (y != 'X') & d['visible']
        xs.append(dv.columns(d['x'], d['shape'], 'C')[g])
        ys.append(y[g])
    return HistGradientBoostingClassifier(**ex2.HGB).fit(np.concatenate(xs), np.concatenate(ys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', default='raw')
    args = parser.parse_args()
    out = ROOT / 'external/oof' / args.labels
    out.mkdir(parents=True, exist_ok=True)
    for fold in range(dv.FOLDS):
        if (out / f'fold{fold}.done').exists():
            continue
        model = train(fold, args.labels)
        views = [('sc', m, 'real', dv.sc_data(m)) for m in dv.SC if dv.fold_of(dv.SC, m) == fold]
        views += [('dfl', m, c, dv.dfl_data(m, c, 'eval')) for m in dv.DFL if dv.fold_of(dv.DFL, m) == fold
                  for c in dg.CONDITIONS]
        for ds, m, c, d in views:
            q = mt.learned_probs(model, dv.columns(d['x'], d['shape'], 'C'))
            np.savez(out / f'{ds}-{m}-{c}.npz', q=q.astype(np.float64), owner=d['owner'],
                     periods=d['periods'], labels=d['labels'])
        (out / f'fold{fold}.done').touch()
        print(f'fold {fold}: {len(views)} views stored', flush=True)
