"""EXPERIMENT-04 runner (v2). One-shot guarded stages; never reads competition data or labels.

    .venv/bin/python -m v2.experiment dev       # development matches only; selects settings
    .venv/bin/python -m v2.experiment holdout   # one time, after dev freeze; no retuning

Both stages refuse to run twice (exclusive .started markers) and the holdout refuses to run if any
bound source, the registration or the frozen v2 model changed after development.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation import pm  # noqa: E402  (official scorer, pinned)
from v2 import idsse, degrade as dg, methods as mt  # noqa: E402

OUT = ROOT / 'work/v2-evaluation'
REGISTRATION = 'work/EXPERIMENT-04.md'
BOUND = ['v2/idsse.py', 'v2/degrade.py', 'v2/methods.py', 'v2/experiment.py', REGISTRATION,
         'possession.py', 'learned_possession.py', 'hybrid_possession.py', 'external_validation.py',
         'first_assessment.py', 'starter-repo/starter/baseline.py',
         'starter-repo/evaluation/possession_metrics.py']
BLEND = [(w, p) for w in (.1, .25, .5, .75, 1.) for p in (0., .1, .25, .5, 1.)]
GRIDS = {'majority': [0, 25, 50, 75, 100, 150, 200, 300],
         'hold_majority': [0, 25, 50, 75, 100, 150, 200, 300],
         'hmm_hand': [.05, .1, .25, .5, 1., 2., 4.],
         'blend_learned': BLEND, 'blend_uniform': BLEND, 'blend_misaligned': BLEND}
# Simplicity order for the candidate tie rule; blend_misaligned is a control and never a candidate.
CANDIDATES = ['majority', 'hold_majority', 'hmm_hand', 'blend_uniform', 'blend_learned']
NON_LEARNING = ['majority', 'hold_majority', 'hmm_hand']
TUNE = 'contested'
FAMILY_TIE, CANDIDATE_TIE = .005, .01
HGB = dict(max_iter=100, max_leaf_nodes=15, min_samples_leaf=50, learning_rate=.08,
           l2_regularization=1., early_stopping=False, random_state=20261028)


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in BOUND}


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return {'spellF1': float(r['spell_f1_mean']), 'macroF1': float(r['macro_f1']),
            'gt_spells': int(r['n_gt_spells']), 'pred_spells': int(r['n_pred_spells'])}


def mapping_check(arrays):
    """Instrument check: nearest-team agreement with BallPossession under 1=home must exceed .5."""
    obs = mt.observe(arrays['players'], arrays['teams'], arrays['ball'], arrays['periods'])
    alive = np.isin(arrays['labels'], ['H', 'A']) & (obs['owner'] >= 0)
    return float(np.mean(obs['owner'][alive] == (arrays['labels'][alive] == 'A')))


def train(matches, loaded):
    xs, ys = [], []
    for m in matches:
        a = loaded[m]
        graded = a['labels'] != 'X'
        for condition in dg.CONDITIONS:
            players, ball = dg.degrade(a, m, condition, 'train')
            xs.append(mt.learned_inputs(players, a['teams'], ball, a['periods'])[graded])
            ys.append(a['labels'][graded])
    return HistGradientBoostingClassifier(**HGB).fit(np.concatenate(xs), np.concatenate(ys))


def predict_all(arrays, match, condition, model, v1, settings):
    """settings: {family: [setting, ...]}. Returns {(method, setting): labels}."""
    periods = arrays['periods']
    players, ball = dg.degrade(arrays, match, condition, 'eval')
    obs = mt.observe(players, arrays['teams'], ball, periods)
    x = mt.learned_inputs(players, arrays['teams'], ball, periods)
    q = mt.learned_probs(model, x)
    sources = {'blend_learned': q, 'blend_uniform': mt.uniform_evidence(len(q)),
               'blend_misaligned': mt.misaligned(q, periods)}
    prior = mt.prior50(obs, periods)
    v1_model, (v1_weight, v1_penalty) = v1
    from hybrid_possession import predict as v1_predict  # frozen v1 hybrid, unchanged
    out = {('baseline50', 50): mt.majority(obs, periods, 50),
           ('v1_frozen', (v1_weight, v1_penalty)): v1_predict(v1_model, x, obs['owner'], periods,
                                                               v1_weight, v1_penalty)}
    for family, grid in settings.items():
        for s in grid:
            if family == 'majority':
                out[(family, s)] = mt.majority(obs, periods, s)
            elif family == 'hold_majority':
                out[(family, s)] = mt.hold_majority(obs, periods, s)
            elif family == 'hmm_hand':
                out[(family, s)] = mt.hmm_hand(obs, periods, s)
            else:
                out[(family, tuple(s))] = mt.blend(sources[family], prior, periods, *s)
    missing = float(np.mean(~np.isfinite(ball[periods > 0]).all(axis=1)))
    return out, missing


def select(rows, family, grid, condition=TUNE):
    """Best mean spellF1 on development; ties within FAMILY_TIE take the earliest grid entry."""
    means = []
    for s in grid:
        vals = [r['spellF1'] for r in rows if r['method'] == family and r['setting'] == _key(s)
                and r['condition'] == condition]
        means.append(float(np.mean(vals)))
    best = max(means)
    i = next(i for i, v in enumerate(means) if v >= best - FAMILY_TIE)
    return grid[i], means[i]


def choose_candidate(family_means):
    best = max(family_means[f] for f in CANDIDATES)
    return next(f for f in CANDIDATES if family_means[f] >= best - CANDIDATE_TIE)


def _key(s):
    return list(s) if isinstance(s, (tuple, list)) else s


def paired(rows, a, b, condition, metric='spellF1'):
    by = {(r['match'], r['method']): r[metric] for r in rows if r['condition'] == condition}
    matches = sorted({m for m, _ in by})
    return [by[(m, a)] - by[(m, b)] for m in matches]


def promotion_gate(rows, candidate):
    d = {c: paired(rows, candidate, 'v1_frozen', c) for c in dg.CONDITIONS}
    macro = paired(rows, candidate, 'v1_frozen', TUNE, 'macroF1')
    checks = {'contested_mean_ge_.02': float(np.mean(d['contested'])) >= .02,
              'contested_wins_ge_3of4': int(np.sum(np.array(d['contested']) > 0)) >= 3,
              'gaps_mean_ge_0': float(np.mean(d['gaps'])) >= 0,
              'natural_mean_ge_-.02': float(np.mean(d['natural'])) >= -.02,
              'contested_macro_mean_ge_-.03': float(np.mean(macro)) >= -.03}
    return {'deltas_vs_v1': d, 'contested_macro_deltas': macro, 'checks': checks,
            'passed': all(checks.values())}


def learned_edge_gate(rows, best_non_learning):
    out = {}
    for control in ['blend_uniform', 'blend_misaligned', best_non_learning]:
        d = paired(rows, 'blend_learned', control, TUNE)
        out[control] = {'deltas': d, 'mean_ge_.02': float(np.mean(d)) >= .02,
                        'wins_ge_3of4': int(np.sum(np.array(d) > 0)) >= 3}
    return {'controls': out, 'passed': all(v['mean_ge_.02'] and v['wins_ge_3of4']
                                           for v in out.values())}


def v1_frozen():
    from first_assessment import frozen_model  # verifies frozen v1 hashes before unpickling
    model, settings, model_hash = frozen_model()
    return (model, tuple(settings)), model_hash


def run_rows(matches, loaded, models, v1, settings, stage):
    rows = []
    for m in matches:
        for condition in dg.CONDITIONS:
            preds, missing = predict_all(loaded[m], m, condition, models[m], v1, settings)
            for (method, s), labels in preds.items():
                rows.append({'stage': stage, 'match': m, 'condition': condition,
                             'method': method, 'setting': _key(s), 'missing_ball': missing,
                             **score(loaded[m]['labels'], labels)})
            print(f'{stage} {m} {condition}: {len(preds)} predictions scored', flush=True)
    return rows


def main(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    (v1, v1_hash) = v1_frozen()
    if stage == 'dev':
        with (OUT / 'dev.started').open('x') as f:
            json.dump({'source_hashes': source_hashes(), 'v1_model_sha256': v1_hash}, f)
        loaded = {m: idsse.load(m) for m in idsse.DEVELOPMENT}
        mapping = {m: mapping_check(loaded[m]) for m in idsse.DEVELOPMENT}
        if min(mapping.values()) <= .5:
            raise SystemExit(f'Label mapping check failed: {mapping}; amend before any outcome.')
        lomo = {m: train([o for o in idsse.DEVELOPMENT if o != m], loaded) for m in idsse.DEVELOPMENT}
        rows = run_rows(idsse.DEVELOPMENT, loaded, lomo, v1, GRIDS, 'development')
        selected = {f: select(rows, f, g) for f, g in GRIDS.items()}
        family_means = {f: v[1] for f, v in selected.items()}
        candidate = choose_candidate(family_means)
        best_non_learning = max(NON_LEARNING, key=lambda f: family_means[f])
        final = train(list(idsse.DEVELOPMENT), loaded)
        with (OUT / 'model.pkl').open('wb') as f:
            pickle.dump(final, f)
        report = {'stage': 'development', 'matches': list(idsse.DEVELOPMENT), 'mapping_check': mapping,
                  'selected': {f: _key(v[0]) for f, v in selected.items()},
                  'dev_means_contested': family_means, 'candidate': candidate,
                  'best_non_learning': best_non_learning, 'rows': rows,
                  'model_sha256': hashlib.sha256((OUT / 'model.pkl').read_bytes()).hexdigest(),
                  'v1_model_sha256': v1_hash, 'source_hashes': source_hashes()}
        (OUT / 'dev.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps({k: report[k] for k in ['selected', 'dev_means_contested', 'candidate',
                                                 'best_non_learning', 'mapping_check']}, indent=1))
        return
    dev = json.loads((OUT / 'dev.json').read_text())
    if dev['source_hashes'] != source_hashes():
        raise SystemExit('Bound source or registration changed after development; amend explicitly.')
    model_path = OUT / 'model.pkl'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != dev['model_sha256']:
        raise SystemExit('v2 model differs from development freeze')
    with (OUT / 'holdout.started').open('x') as f:
        json.dump({'source_hashes': source_hashes(), 'model_sha256': dev['model_sha256']}, f)
    with model_path.open('rb') as f:
        model = pickle.load(f)  # locally trained in the dev stage and hash-verified above
    loaded = {m: idsse.load(m) for m in idsse.HOLDOUT}
    mapping = {m: mapping_check(loaded[m]) for m in idsse.HOLDOUT}
    if min(mapping.values()) <= .5:
        raise SystemExit(f'Label mapping check failed on holdout: {mapping}')
    settings = {f: [s] for f, s in dev['selected'].items()}
    rows = run_rows(idsse.HOLDOUT, loaded, {m: model for m in idsse.HOLDOUT}, v1, settings, 'holdout')
    report = {'stage': 'one-time holdout', 'matches': list(idsse.HOLDOUT), 'mapping_check': mapping,
              'selected': dev['selected'], 'candidate': dev['candidate'], 'rows': rows,
              'promotion_gate': promotion_gate(rows, dev['candidate']),
              'learned_edge_gate': learned_edge_gate(rows, dev['best_non_learning']),
              'source_hashes': source_hashes(), 'model_sha256': dev['model_sha256']}
    (OUT / 'holdout.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ['candidate', 'promotion_gate', 'learned_edge_gate']},
                     indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'holdout'])
    main(parser.parse_args().stage)
