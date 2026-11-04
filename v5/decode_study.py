"""EXPERIMENT-08 decoding study on stored out-of-fold probabilities (see work/EXPERIMENT-08.md).

    OMP_NUM_THREADS=1 .venv/bin/python -m v5.decode_study

Writes work/v5-development/{rows.jsonl,selection.json}.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import experiment as ex2, methods as mt  # noqa: E402
from v3.decode import viterbi2  # noqa: E402
from v5.labels import relabel  # noqa: E402

OOF = ROOT / 'external/oof'
OUT = ROOT / 'work/v5-development'
CONVENTIONS = ['raw', 'controlled1', 'controlled2', 'controlled3']
TRAINING = ['raw', 'controlled2']
WS = [.75, 1.]
UNIFORM = [.25, .5, 1., 2., 4.]
ASYM = [(t, d) for t in (.25, .5, 1., 2.) for d in (.5, 1., 2., 4.)]
REFERENCE = ('raw', 'U', 1., (.5,))


def candidates():
    for tr in TRAINING:
        for w in WS:
            for p in UNIFORM:
                yield (tr, 'U', w, (p,))
            for t, d in ASYM:
                yield (tr, 'S', w, (t, d))


def decode(q, prior, periods, kind, w, params):
    mix = w * q + (1 - w) * prior
    costs = -np.log(np.clip(mix, 1e-6, 1.)) / 25.
    return mt.viterbi(costs, periods, params[0]) if kind == 'U' else viterbi2(costs, periods, *params)


def view(name):
    out = []
    data = {tr: np.load(OOF / tr / name) for tr in TRAINING}
    base = data['raw']
    periods, labels = base['periods'], base['labels']
    prior = mt.prior50({'owner': base['owner']}, periods)
    truths = {c: relabel(labels, periods, c) for c in CONVENTIONS}
    for tr, kind, w, params in candidates():
        pred = decode(data[tr]['q'], prior, periods, kind, w, params)
        scores = {c: ex2.score(truths[c], pred) for c in CONVENTIONS}
        out.append({'view': name, 'training': tr, 'kind': kind, 'w': w, 'params': list(params),
                    **{f'spell_{c}': scores[c]['spellF1'] for c in CONVENTIONS},
                    'macro_raw': scores['raw']['macroF1'], 'pred_spells': scores['raw']['pred_spells']})
    return out


def key(r):
    return (r['training'], r['kind'], r['w'], tuple(r['params']))


def mean_conv(r):
    return float(np.mean([r[f'spell_{c}'] for c in CONVENTIONS]))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (OOF / 'raw').glob('*.npz'))
    assert all((OOF / 'controlled2' / n).exists() for n in names), 'missing controlled2 OOF files'
    with ProcessPoolExecutor(4) as pool:
        rows = [r for chunk in pool.map(view, names) for r in chunk]
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    def primary(k):
        return float(np.mean([mean_conv(r) for r in by[k] if r['view'].startswith('sc-')]))
    order = list(candidates())
    best = max(primary(k) for k in order)
    selected = next(k for k in order if primary(k) >= best - .005)
    def per_view(k, prefix):
        return {r['view']: mean_conv(r) for r in by[k] if r['view'].startswith(prefix)}
    sel_sc, ref_sc = per_view(selected, 'sc-'), per_view(REFERENCE, 'sc-')
    d_sc = [sel_sc[v] - ref_sc[v] for v in sorted(sel_sc)]
    def dfl(k, cond):
        return {r['view']: mean_conv(r) for r in by[k] if r['view'].startswith('dfl-') and r['view'].endswith(f'-{cond}.npz')}
    d_con = [dfl(selected, 'contested')[v] - dfl(REFERENCE, 'contested')[v] for v in dfl(selected, 'contested')]
    d_nat = [dfl(selected, 'natural')[v] - dfl(REFERENCE, 'natural')[v] for v in dfl(selected, 'natural')]
    macro = [np.mean([r['macro_raw'] for r in by[selected] if r['view'].startswith('sc-')]) -
             np.mean([r['macro_raw'] for r in by[REFERENCE] if r['view'].startswith('sc-')])]
    checks = {'primary_delta_ge_.02': float(np.mean(d_sc)) >= .02,
              'sc_wins_ge_14of20': int(np.sum(np.array(d_sc) > 0)) >= 14,
              'dfl_contested_ge_-.01': float(np.mean(d_con)) >= -.01,
              'dfl_natural_ge_-.02': float(np.mean(d_nat)) >= -.02,
              'sc_macro_raw_ge_-.02': float(macro[0]) >= -.02}
    report = {'selected': list(selected[:3]) + [list(selected[3])], 'reference': list(REFERENCE[:3]) + [list(REFERENCE[3])],
              'primary_selected': primary(selected), 'primary_reference': primary(REFERENCE),
              'sc_deltas': d_sc, 'dfl_contested_deltas': d_con, 'dfl_natural_deltas': d_nat,
              'sc_macro_raw_delta': float(macro[0]), 'checks': checks, 'passed': all(checks.values()),
              'top10': sorted(((primary(k), list(k[:3]) + [list(k[3])]) for k in order), reverse=True)[:10]}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ['selected', 'primary_selected', 'primary_reference', 'checks', 'passed']}, indent=1))
