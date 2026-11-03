"""EXPERIMENT-05 Part B: one-time holdout on 19 SkillCorner matches (real broadcast tracking).

    OMP_NUM_THREADS=4 .venv/bin/python -m v3.holdout

Refuses to run twice, or if the frozen v3 model or any bound source changed since selection.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import experiment as ex2, methods as mt  # noqa: E402
from v3 import develop as dv, features as f3, skillcorner as sc  # noqa: E402

OUT = ROOT / 'work/v3-evaluation'
BOUND = ['v3/features.py', 'v3/decode.py', 'v3/develop.py', 'v3/skillcorner.py', 'v3/holdout.py',
         'work/EXPERIMENT-05.md', 'work/v3-development/selection.json']
METHODS = ['v3', 'v2_frozen', 'v1_frozen', 'baseline50', 'hold_majority']


def bound_hashes():
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in BOUND}
    hashes.update(ex2.source_hashes())
    return hashes


def frozen_models():
    from v2.verified_inference import check_v2_freeze, runtime
    folder = ROOT / 'work/v2-evaluation'
    dev, held, prov = [json.loads((folder / f'{n}.json').read_text()) for n in ['dev', 'holdout', 'provenance']]
    raw = (folder / 'model.pkl').read_bytes()
    v2_settings = check_v2_freeze(dev, held, prov, hashlib.sha256(raw).hexdigest(),
                                  ex2.source_hashes(), runtime())
    v1, _ = ex2.v1_frozen()
    sel = json.loads((ROOT / 'work/v3-development/selection.json').read_text())
    raw3 = (ROOT / 'work/v3-development/model.pkl').read_bytes()
    if sel['variant'] == 'A' or hashlib.sha256(raw3).hexdigest() != sel['model_sha256']:
        raise SystemExit('No frozen v3 candidate, or model hash mismatch')
    for p, h in sel['source_hashes'].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            raise SystemExit(f'{p} changed since v3 selection')
    # All three pickles were trained by this project and are hash-verified above.
    return pickle.loads(raw), v2_settings, v1, pickle.loads(raw3), sel


def predict(a, v2_model, v2_settings, v1, v3_model, sel):
    per = a['periods']
    obs = mt.observe(a['players'], a['teams'], a['ball'], per)
    x = f3.build(a['players'], a['teams'], a['ball'], per)
    prior = mt.prior50(obs, per)
    setting = sel['best'][sel['variant']]['setting']
    q3 = mt.learned_probs(v3_model, x)
    v3 = dv.decode_d(q3, prior, per, *setting) if sel['variant'] == 'D' else mt.blend(q3, prior, per, *setting)
    from hybrid_possession import predict as v1_predict
    v1_model, (w1, p1) = v1
    return {'v3': v3,
            'v2_frozen': mt.blend(mt.learned_probs(v2_model, x[:, :8]), prior, per, *v2_settings),
            'v1_frozen': v1_predict(v1_model, x[:, :8], obs['owner'], per, w1, p1),
            'baseline50': mt.majority(obs, per, 50),
            'hold_majority': mt.hold_majority(obs, per, 50)}


def wins(rows, a, b, metric='spellF1', labels='tracking'):
    by = {(r['match'], r['method']): r[metric] for r in rows if r['labels'] == labels}
    d = [by[(m, a)] - by[(m, b)] for m in sorted({m for m, _ in by})]
    return {'deltas': d, 'mean': float(np.mean(d)), 'wins': int(np.sum(np.array(d) > 0)), 'n': len(d)}


def gates(rows):
    g1s, g1m = wins(rows, 'v3', 'v2_frozen'), wins(rows, 'v3', 'v2_frozen', 'macroF1')
    g1 = {'spell': g1s, 'macro': g1m,
          'passed': g1s['mean'] > 0 and g1s['wins'] >= 14 and g1m['mean'] >= -.03}
    t = {c: wins(rows, 'v2_frozen', c) for c in ['baseline50', 'hold_majority']}
    return {'G1_v3_replaces_v2': g1,
            'T_v2_transfer': {**t, 'confirmed': all(v['wins'] >= 14 for v in t.values())},
            'sensitivity_phase_labels_v3_vs_v2': wins(rows, 'v3', 'v2_frozen', labels='phases')}


if __name__ == '__main__':
    models = frozen_models()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'holdout.started').open('x') as f:
        json.dump({'bound_hashes': bound_hashes()}, f)
    rows, excluded = [], {}
    for match in sc.match_ids():
        a = sc.load(match, with_phases=True)
        check = ex2.mapping_check(a)
        if check <= .5:
            excluded[match] = check
            continue
        preds = predict(a, *models)
        missing = float(np.mean(~np.isfinite(a['ball'][a['periods'] > 0]).all(axis=1)))
        for method, labels in preds.items():
            for kind, truth in [('tracking', a['labels']), ('phases', a['phase_labels'])]:
                rows.append({'match': match, 'method': method, 'labels': kind, 'mapping_check': check,
                             'missing_ball': missing, **ex2.score(truth, labels)})
        print(f'holdout {match}: scored', flush=True)
    if len(excluded) > 3:
        raise SystemExit(f'More than 3 matches failed the instrument check: {excluded}')
    report = {'stage': 'one-time SkillCorner holdout', 'excluded': excluded, 'rows': rows,
              'gates': gates(rows), 'bound_hashes': bound_hashes()}
    (OUT / 'holdout.json').write_text(json.dumps(report, indent=1) + '\n')
    summary = {m: float(np.mean([r['spellF1'] for r in rows if r['method'] == m and r['labels'] == 'tracking']))
               for m in METHODS}
    print(json.dumps({'mean_spellF1': summary, 'excluded': excluded,
                      'G1': report['gates']['G1_v3_replaces_v2']['passed'],
                      'T': report['gates']['T_v2_transfer']['confirmed']}, indent=1))
