"""EXPERIMENT-10: out-of-fold variant D (v4 features + control) vs stored v4-recipe OOF (C).

    OMP_NUM_THREADS=4 .venv/bin/python -m v6.develop

Writes external/oof/control/*.npz and work/v6-development/{rows.jsonl,selection.json}.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, idsse, methods as mt  # noqa: E402
from v3 import skillcorner as sc  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v5.labels import relabel  # noqa: E402
from v6.control import control_features  # noqa: E402

CACHE = ROOT / 'external/control-cache'
OOF_C, OOF_D = ROOT / 'external/oof/raw', ROOT / 'external/oof/control'
OUT = ROOT / 'work/v6-development'
CONVENTIONS = ('raw', 'controlled1', 'controlled2', 'controlled3')


def ctrl(ds, match, condition, purpose):
    path = CACHE / f'{ds}-{match}-{condition}-{purpose}.npy'
    if path.exists():
        return np.load(path)
    if ds == 'dfl':
        a = idsse.load(match)
        players, ball = dg.degrade(a, match, condition, purpose)
        x = control_features(players, a['teams'], ball, a['periods'])
    else:
        a = sc.load(match)
        x = control_features(a['players'], a['teams'], a['ball'], a['periods'])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def features(ds, match, condition, purpose, d):
    return np.hstack([dv.columns(d['x'], d['shape'], 'C'), ctrl(ds, match, condition, purpose)])


def train(fold):
    xs, ys = [], []
    for m in dv.DFL:
        if dv.fold_of(dv.DFL, m) != fold:
            for c in dg.CONDITIONS:
                d = dv.dfl_data(m, c, 'train')
                g = d['labels'] != 'X'
                xs.append(features('dfl', m, c, 'train', d)[g]); ys.append(d['labels'][g])
    for m in dv.SC:
        if dv.fold_of(dv.SC, m) != fold:
            d = dv.sc_data(m)
            g = (d['labels'] != 'X') & d['visible']
            xs.append(features('sc', m, 'real', 'eval', d)[g]); ys.append(d['labels'][g])
    return HistGradientBoostingClassifier(**ex2.HGB).fit(np.concatenate(xs), np.concatenate(ys))


def score_view(name):
    c, d = np.load(OOF_C / name), np.load(OOF_D / name)
    periods, labels = c['periods'], c['labels']
    prior = mt.prior50({'owner': c['owner']}, periods)
    truths = {k: relabel(labels, periods, k) for k in CONVENTIONS}
    row = {'view': name}
    for tag, q in (('C', c['q']), ('D', d['q'])):
        pred = mt.blend(q, prior, periods, 1., .5)
        s = {k: ex2.score(truths[k], pred) for k in CONVENTIONS}
        row[tag] = float(np.mean([s[k]['spellF1'] for k in CONVENTIONS]))
        row[f'{tag}_macro_raw'] = s['raw']['macroF1']
        row[f'{tag}_spells'] = s['raw']['pred_spells']
    return row


if __name__ == '__main__':
    OOF_D.mkdir(parents=True, exist_ok=True)
    for fold in range(dv.FOLDS):
        if (OOF_D / f'fold{fold}.done').exists():
            continue
        model = train(fold)
        views = [('sc', m, 'real', 'eval', dv.sc_data(m)) for m in dv.SC if dv.fold_of(dv.SC, m) == fold]
        views += [('dfl', m, c, 'eval', dv.dfl_data(m, c, 'eval')) for m in dv.DFL
                  if dv.fold_of(dv.DFL, m) == fold for c in dg.CONDITIONS]
        for ds, m, c, p, d in views:
            q = mt.learned_probs(model, features(ds, m, c, p, d))
            np.savez(OOF_D / f'{ds}-{m}-{c}.npz', q=q, owner=d['owner'], periods=d['periods'], labels=d['labels'])
        (OOF_D / f'fold{fold}.done').touch()
        print(f'fold {fold}: {len(views)} views', flush=True)
    from concurrent.futures import ProcessPoolExecutor
    names = sorted(p.name for p in OOF_C.glob('*.npz'))
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(score_view, names))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    def deltas(pred):
        return [r['D'] - r['C'] for r in rows if pred(r['view'])]
    sc_d = deltas(lambda v: v.startswith('sc-'))
    con = deltas(lambda v: v.startswith('dfl-') and v.endswith('-contested.npz'))
    nat = deltas(lambda v: v.startswith('dfl-') and v.endswith('-natural.npz'))
    macro = float(np.mean([r['D_macro_raw'] - r['C_macro_raw'] for r in rows if r['view'].startswith('sc-')]))
    checks = {'sc_mean_ge_.015': float(np.mean(sc_d)) >= .015, 'sc_wins_ge_14of20': int(np.sum(np.array(sc_d) > 0)) >= 14,
              'dfl_contested_ge_-.005': float(np.mean(con)) >= -.005, 'dfl_natural_ge_-.02': float(np.mean(nat)) >= -.02,
              'sc_macro_ge_-.02': macro >= -.02}
    summary = {'sc_C': float(np.mean([r['C'] for r in rows if r['view'].startswith('sc-')])),
               'sc_D': float(np.mean([r['D'] for r in rows if r['view'].startswith('sc-')])),
               'sc_delta': float(np.mean(sc_d)), 'sc_wins': int(np.sum(np.array(sc_d) > 0)),
               'dfl_contested_delta': float(np.mean(con)), 'dfl_natural_delta': float(np.mean(nat)),
               'sc_macro_delta': macro, 'checks': checks, 'passed': all(checks.values())}
    (OUT / 'selection.json').write_text(json.dumps({**summary, 'sc_deltas': sc_d, 'dfl_contested_deltas': con,
                                                    'dfl_natural_deltas': nat}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
