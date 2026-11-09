"""EXPERIMENT-13 analogue: broadcast-style ball fusion on DFL contested views, out of fold.

    OMP_NUM_THREADS=4 .venv/bin/python -m v8.analogue

Retrains the EXPERIMENT-10 fold models (deterministic, cached in external/models/) and scores each
DFL match on its contested input (reference) and on the same input with the ball filled from an
aligned simulated second view. Resumable: per-match rows are cached in work/v8-analogue/rows/.
Writes work/v8-analogue/{rows.jsonl,selection.json}.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, idsse, methods as mt  # noqa: E402
from v3 import features as f3  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v4.shape import team_shape  # noqa: E402
from v5.labels import relabel  # noqa: E402
from v6 import develop as d6  # noqa: E402
from v6.control import control_features  # noqa: E402
from v6.infer import predict  # noqa: E402
from v8 import align as al  # noqa: E402
from v8.simulate import second_view  # noqa: E402

OUT = ROOT / 'work/v8-analogue'
MODELS = ROOT / 'external/models'
CONVENTIONS = ('raw', 'controlled1', 'controlled2', 'controlled3')


def inputs(players, teams, ball, periods):
    obs = mt.observe(players, teams, ball, periods)
    return (f3.build(players, teams, ball, periods), team_shape(players, teams, periods)[0],
            control_features(players, teams, ball, periods), obs['owner'])


def fold_model(fold):
    x, y = [], []
    for m in dv.DFL:
        if dv.fold_of(dv.DFL, m) != fold:
            for c in dg.CONDITIONS:
                d = dv.dfl_data(m, c, 'train')
                g = d['labels'] != 'X'
                x.append(d6.features('dfl', m, c, 'train', d)[g]); y.append(d['labels'][g])
    for m in dv.SC:
        if dv.fold_of(dv.SC, m) != fold:
            d = dv.sc_data(m)
            g = (d['labels'] != 'X') & d['visible']
            x.append(d6.features('sc', m, 'real', 'eval', d)[g]); y.append(d['labels'][g])
    return HistGradientBoostingClassifier(**ex2.HGB).fit(np.concatenate(x), np.concatenate(y))


def cached_model(fold):
    """EXPERIMENT-10 recipe fold model, trained once and pickled (deterministic given the data)."""
    path = MODELS / f'v6-recipe-fold{fold}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    model = fold_model(fold)
    MODELS.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    return model


if __name__ == '__main__':
    (OUT / 'rows').mkdir(parents=True, exist_ok=True)
    rows = []
    for m in dv.DFL:
        cached = OUT / 'rows' / f'{m}.json'
        if cached.exists():
            rows.append(json.loads(cached.read_text()))
            continue
        model = cached_model(dv.fold_of(dv.DFL, m))
        a = idsse.load(m)
        periods, labels, teams = a['periods'], a['labels'], a['teams']
        tp, tb = dg.degrade(a, m, 'contested', 'eval')
        seed = int(hashlib.sha256(f'{m}|analogue|20261104'.encode()).hexdigest()[:8], 16)
        bp, bb, truth = second_view(a['players'], a['ball'], periods, delta=1234, seed=seed)
        res = al.align(tp, bp, periods)
        fused, source = al.fuse_ball(tb, bb, res)
        truths = {c: relabel(labels, periods, c) for c in CONVENTIONS}
        row = {'match': m, 'flip': list(res['flip']), 'offsets': res['offsets'],
               'missing_before': float(np.mean(~np.isfinite(tb[periods > 0]).all(axis=1))),
               'missing_after': float(np.mean(source[periods > 0] == -1)),
               'alignment_exact_share': float(np.mean(res['mapping'][truth >= 0][res['mapping'][truth >= 0] >= 0]
                                                      == truth[truth >= 0][res['mapping'][truth >= 0] >= 0]))}
        for tag, ball in (('reference', tb), ('fused', fused)):
            pred = predict(model, *inputs(tp, teams, ball, periods), periods)
            s = {c: ex2.score(truths[c], pred) for c in CONVENTIONS}
            row[tag] = float(np.mean([s[c]['spellF1'] for c in CONVENTIONS]))
            row[f'{tag}_macro'] = s['raw']['macroF1']
        rows.append(row)
        cached.write_text(json.dumps(row) + '\n')
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'offsets'}, flush=True)
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    d = [r['fused'] - r['reference'] for r in rows]
    dm = [r['fused_macro'] - r['reference_macro'] for r in rows]
    checks = {'contested_mean_ge_.02': float(np.mean(d)) >= .02, 'wins_ge_6of7': int(np.sum(np.array(d) > 0)) >= 6,
              'macro_ge_0': float(np.mean(dm)) >= 0}
    report = {'deltas': d, 'macro_deltas': dm, 'mean_delta': float(np.mean(d)),
              'missing_before': float(np.mean([r['missing_before'] for r in rows])),
              'missing_after': float(np.mean([r['missing_after'] for r in rows])),
              'checks': checks, 'passed': all(checks.values())}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
