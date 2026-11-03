"""EXPERIMENT-06: match-grouped 5-fold cross-validation of variants A/B/C vs frozen v2.

    OMP_NUM_THREADS=4 .venv/bin/python -m v4.develop

Writes work/v4-development/{rows.jsonl,selection.json}. Rows are appended per fold, so an
interrupted run resumes at the next unfinished fold.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, idsse, methods as mt  # noqa: E402
from v3 import develop as dv3, skillcorner as sc  # noqa: E402
from v4.shape import team_shape  # noqa: E402

OUT = ROOT / 'work/v4-development'
DFL = tuple(sorted(idsse.DEVELOPMENT + idsse.HOLDOUT))
# Import-safe without the public SkillCorner files (e.g. inference-only checkouts).
SC = tuple(sorted([sc.PILOT] + sc.match_ids())) if (sc.META.parent / 'matches.json').exists() else ()
FOLDS = 5
GRID = [(w, p) for w in (.5, .75, 1.) for p in (.1, .25, .5, 1.)]
VARIANTS = ['A', 'B', 'C']
FAMILY_TIE, VARIANT_TIE = .005, .01


def columns(x37, shape, variant):
    if variant == 'A':
        return x37[:, :8]
    if variant == 'B':
        return np.hstack([x37[:, :8], shape])
    return np.hstack([x37, shape])


def dfl_data(match, condition, purpose):
    path = ROOT / 'external/idsse/cache/v4' / f'{match}-{condition}-{purpose}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    d = dv3.frame_data(match, condition, purpose)
    a = idsse.load(match)
    players, _ = dg.degrade(a, match, condition, purpose)
    d['shape'] = team_shape(players, a['teams'], a['periods'])[0]
    d['visible'] = np.isfinite(d['x'][:, 2])  # ball speed is finite only when the ball is visible
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return d


def sc_data(match):
    path = ROOT / 'external/skillcorner-lfs/cache' / f'{match}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    a = sc.load(match)
    from v3 import features as f3
    obs = mt.observe(a['players'], a['teams'], a['ball'], a['periods'])
    d = {'x': f3.build(a['players'], a['teams'], a['ball'], a['periods']),
         'shape': team_shape(a['players'], a['teams'], a['periods'])[0], 'owner': obs['owner'],
         'visible': obs['visible'], 'periods': a['periods'], 'labels': a['labels']}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return d


def fold_of(ids, match):
    return sorted(ids).index(match) % FOLDS


def training_set(fold, variant):
    xs, ys = [], []
    for m in DFL:
        if fold_of(DFL, m) == fold:
            continue
        for c in dg.CONDITIONS:
            d = dfl_data(m, c, 'train')
            g = d['labels'] != 'X'
            xs.append(columns(d['x'], d['shape'], variant)[g])
            ys.append(d['labels'][g])
    for m in SC:
        if fold_of(SC, m) == fold:
            continue
        d = sc_data(m)
        g = (d['labels'] != 'X') & d['visible']  # registered: SkillCorner visible-ball frames only
        xs.append(columns(d['x'], d['shape'], variant)[g])
        ys.append(d['labels'][g])
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(d, models, v2_model):
    per, gt = d['periods'], d['labels']
    prior = mt.prior50({'owner': d['owner']}, per)
    out = {('v2_frozen', (.75, .5)): mt.blend(mt.learned_probs(v2_model, d['x'][:, :8]), prior, per, .75, .5),
           ('baseline50', 50): mt.majority({'owner': d['owner']}, per, 50)}
    for v, model in models.items():
        q = mt.learned_probs(model, columns(d['x'], d['shape'], v))
        for w, p in GRID:
            out[(v, (w, p))] = mt.blend(q, prior, per, w, p)
    return {k: ex2.score(gt, labels) for k, labels in out.items()}


def select(rows):
    best = {}
    for v in VARIANTS:
        means = []
        for s in GRID:
            vals = [r['spellF1'] for r in rows if r['dataset'] == 'sc' and r['method'] == v and r['setting'] == list(s)]
            means.append(float(np.mean(vals)))
        top = max(means)
        i = next(i for i, m in enumerate(means) if m >= top - FAMILY_TIE)
        best[v] = (list(GRID[i]), means[i])
    top = max(m for _, m in best.values())
    return next(v for v in VARIANTS if best[v][1] >= top - VARIANT_TIE), best


def gate(rows, variant, setting):
    def deltas(dataset, condition, metric='spellF1'):
        by = {}
        for r in rows:
            if r['dataset'] == dataset and r['condition'] == condition:
                if r['method'] == variant and r['setting'] == setting:
                    by.setdefault(r['match'], {})['v'] = r[metric]
                elif r['method'] == 'v2_frozen':
                    by.setdefault(r['match'], {})['r'] = r[metric]
        return [x['v'] - x['r'] for _, x in sorted(by.items())]
    s, sm = deltas('sc', 'real'), deltas('sc', 'real', 'macroF1')
    dc, dn = deltas('dfl', 'contested'), deltas('dfl', 'natural')
    checks = {'sc_mean_ge_.03': float(np.mean(s)) >= .03, 'sc_wins_ge_15of20': int(np.sum(np.array(s) > 0)) >= 15,
              'sc_macro_mean_ge_-.02': float(np.mean(sm)) >= -.02, 'dfl_contested_mean_ge_0': float(np.mean(dc)) >= 0,
              'dfl_natural_mean_ge_-.02': float(np.mean(dn)) >= -.02}
    return {'sc_deltas': s, 'sc_macro_deltas': sm, 'dfl_contested_deltas': dc, 'dfl_natural_deltas': dn,
            'checks': checks, 'passed': all(checks.values())}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows_file = OUT / 'rows.jsonl'
    done = {json.loads(l)['fold'] for l in rows_file.read_text().splitlines()} if rows_file.exists() else set()
    v2_model = pickle.loads((ROOT / 'work/v2-evaluation/model.pkl').read_bytes())  # hash-checked by tests
    for fold in range(FOLDS):
        if fold in done:
            continue
        models = {}
        for v in VARIANTS:
            x, y = training_set(fold, v)
            models[v] = HistGradientBoostingClassifier(**ex2.HGB).fit(x, y)
            print(f'fold {fold} variant {v}: trained on {len(y)} frames', flush=True)
        lines = []
        for m in SC:
            if fold_of(SC, m) == fold:
                for (method, s), r in evaluate(sc_data(m), models, v2_model).items():
                    lines.append({'fold': fold, 'dataset': 'sc', 'match': str(m), 'condition': 'real',
                                  'method': method, 'setting': list(s) if isinstance(s, tuple) else s, **r})
        for m in DFL:
            if fold_of(DFL, m) == fold:
                for c in dg.CONDITIONS:
                    for (method, s), r in evaluate(dfl_data(m, c, 'eval'), models, v2_model).items():
                        lines.append({'fold': fold, 'dataset': 'dfl', 'match': m, 'condition': c,
                                      'method': method, 'setting': list(s) if isinstance(s, tuple) else s, **r})
        with rows_file.open('a') as f:
            f.writelines(json.dumps(r) + '\n' for r in lines)
        print(f'fold {fold}: {len(lines)} rows scored', flush=True)
    rows = [json.loads(l) for l in rows_file.read_text().splitlines()]
    variant, best = select(rows)
    report = {'variant': variant, 'best': {v: {'setting': s, 'sc_mean': m} for v, (s, m) in best.items()},
              'gate': gate(rows, variant, best[variant][0])}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'variant': variant, 'best': report['best'], 'checks': report['gate']['checks'],
                      'passed': report['gate']['passed']}, indent=1))
