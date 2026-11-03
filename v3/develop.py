"""EXPERIMENT-05 Part A: leave-one-match-out development over all 7 DFL matches (diagnostic only).

    OMP_NUM_THREADS=4 .venv/bin/python -m v3.develop

Writes work/v3-development/{rows.jsonl,selection.json,model.pkl}. Selection and the final all-7 fit
follow the registered rule; no SkillCorner data is read here.
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
from v3 import decode as d3, features as f3  # noqa: E402

OUT = ROOT / 'work/v3-development'
CACHE = ROOT / 'external/idsse/cache/v3'
MATCHES = tuple(sorted(idsse.DEVELOPMENT + idsse.HOLDOUT))
HGB_V2 = dict(ex2.HGB)
HGB_BIG = dict(max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, learning_rate=.06,
               l2_regularization=1., early_stopping=False, random_state=20261103)
GRID_U = list(ex2.BLEND)
GRID_D = [(w, t, d) for w in (.5, .75, 1.) for t in (.25, .5, 1., 2.) for d in (.1, .25, .5, 1.)]
ORDER = ['A', 'B', 'C', 'D']
TIE = .01


def frame_data(match, condition, purpose):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{match}-{condition}-{purpose}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    a = idsse.load(match)
    players, ball = dg.degrade(a, match, condition, purpose)
    obs = mt.observe(players, a['teams'], ball, a['periods'])
    out = {'x': f3.build(players, a['teams'], ball, a['periods']), 'owner': obs['owner'],
           'periods': a['periods'], 'labels': a['labels']}
    np.savez(path, **out)
    return out


def fit(params, xs, ys):
    return HistGradientBoostingClassifier(**params).fit(np.concatenate(xs), np.concatenate(ys))


def train_models(matches):
    xs, ys = [], []
    for m in matches:
        for c in dg.CONDITIONS:
            d = frame_data(m, c, 'train')
            graded = d['labels'] != 'X'
            xs.append(d['x'][graded])
            ys.append(d['labels'][graded])
    return {'v1': fit(HGB_V2, [x[:, :8] for x in xs], ys), 'v3': fit(HGB_V2, xs, ys),
            'v3big': fit(HGB_BIG, xs, ys)}


def prior(d):
    return mt.prior50({'owner': d['owner']}, d['periods'])


def decode_d(q, pr, periods, w, t, dead):
    mix = w * q + (1 - w) * pr
    return d3.viterbi2(-np.log(np.clip(mix, 1e-6, 1.)) / 25., periods, t, dead)


def evaluate_fold(models, match, v2_model, rows_file):
    for c in dg.CONDITIONS:
        d = frame_data(match, c, 'eval')
        pr, per, gt = prior(d), d['periods'], d['labels']
        q = {k: mt.learned_probs(m, d['x'][:, :8] if k == 'v1' else d['x']) for k, m in models.items()}
        preds = {('baseline50', 50): mt.majority({'owner': d['owner']}, per, 50),
                 ('v2_frozen_in_sample_for_dev3', (.75, .5)):
                     mt.blend(mt.learned_probs(v2_model, d['x'][:, :8]), pr, per, .75, .5)}
        for variant, key in [('A', 'v1'), ('B', 'v3'), ('C', 'v3big')]:
            for w, p in GRID_U:
                preds[(variant, (w, p))] = mt.blend(q[key], pr, per, w, p)
        for w, t, dead in GRID_D:
            preds[('D', (w, t, dead))] = decode_d(q['v3big'], pr, per, w, t, dead)
        with rows_file.open('a') as f:
            for (method, s), labels in preds.items():
                f.write(json.dumps({'match': match, 'condition': c, 'method': method,
                                    'setting': list(s) if isinstance(s, tuple) else s,
                                    **ex2.score(gt, labels)}) + '\n')
        print(f'fold {match} {c}: {len(preds)} scored', flush=True)


def select(rows):
    best = {}
    for v in ORDER:
        settings = sorted({json.dumps(r['setting']) for r in rows if r['method'] == v})
        means = {s: float(np.mean([r['spellF1'] for r in rows if r['method'] == v and
                                   json.dumps(r['setting']) == s and r['condition'] == 'contested']))
                 for s in settings}
        grid = [json.dumps(list(g)) for g in (GRID_D if v == 'D' else GRID_U)]
        top = max(means.values())
        chosen = next(s for s in grid if means[s] >= top - .005)
        best[v] = (json.loads(chosen), means[chosen])
    top = max(m for _, m in best.values())
    variant = next(v for v in ORDER if best[v][1] >= top - TIE)
    return variant, best


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows_file = OUT / 'rows.jsonl'
    done = set()
    if rows_file.exists():
        done = {json.loads(l)['match'] for l in rows_file.read_text().splitlines()}
    v2_model = pickle.loads((ROOT / 'work/v2-evaluation/model.pkl').read_bytes())  # hash-checked by tests
    for m in MATCHES:
        if m in done:
            continue
        evaluate_fold(train_models([o for o in MATCHES if o != m]), m, v2_model, rows_file)
    rows = [json.loads(l) for l in rows_file.read_text().splitlines()]
    variant, best = select(rows)
    report = {'variant': variant, 'best': {v: {'setting': s, 'contested_mean': m}
                                           for v, (s, m) in best.items()}}
    if variant != 'A':
        params, cols = (HGB_V2, 37) if variant == 'B' else (HGB_BIG, 37)
        xs, ys = [], []
        for m in MATCHES:
            for c in dg.CONDITIONS:
                d = frame_data(m, c, 'train')
                g = d['labels'] != 'X'
                xs.append(d['x'][g][:, :cols])
                ys.append(d['labels'][g])
        model = fit(params, xs, ys)
        (OUT / 'model.pkl').write_bytes(pickle.dumps(model))
        report['model_sha256'] = hashlib.sha256((OUT / 'model.pkl').read_bytes()).hexdigest()
        report['source_hashes'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                   for p in ['v3/features.py', 'v3/decode.py', 'v3/develop.py']}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
