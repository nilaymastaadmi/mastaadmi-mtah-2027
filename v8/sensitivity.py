"""EXPERIMENT-16: noise sensitivity of the EXPERIMENT-13 analogue (diagnostic only).

    OMP_NUM_THREADS=1 .venv/bin/python -m v8.sensitivity [--arms N050 N030 N080]

Same procedure as v8/analogue.py, with the simulated second view's position noise set per arm.
Resumable: rows cached in work/v8-sensitivity/<arm>/rows/<match>.json. Writes summary.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, idsse  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v5.labels import relabel  # noqa: E402
from v6.infer import predict  # noqa: E402
from v8 import align as al  # noqa: E402
from v8.analogue import CONVENTIONS, cached_model, inputs  # noqa: E402
from v8.simulate import second_view  # noqa: E402

OUT = ROOT / 'work/v8-sensitivity'
ARMS = {'N030': .3, 'N050': .5, 'N080': .8}


def run_match(m, noise):
    model = cached_model(dv.fold_of(dv.DFL, m))
    a = idsse.load(m)
    periods, labels, teams = a['periods'], a['labels'], a['teams']
    tp, tb = dg.degrade(a, m, 'contested', 'eval')
    seed = int(hashlib.sha256(f'{m}|analogue|20261104'.encode()).hexdigest()[:8], 16)  # as EXPERIMENT-13
    bp, bb, truth = second_view(a['players'], a['ball'], periods, delta=1234, seed=seed, noise=noise)
    res = al.align(tp, bp, periods)
    fused, source = al.fuse_ball(tb, bb, res)
    truths = {c: relabel(labels, periods, c) for c in CONVENTIONS}
    mp = res['mapping']
    ok = (truth >= 0) & (mp >= 0)
    row = {'match': m, 'noise': noise, 'flip': list(res['flip']),
           'accepted_share': float(np.mean(mp >= 0)),
           'missing_before': float(np.mean(~np.isfinite(tb[periods > 0]).all(axis=1))),
           'missing_after': float(np.mean(source[periods > 0] == -1)),
           'alignment_exact_share': float(np.mean(mp[ok] == truth[ok])) if ok.any() else None}
    for tag, ball in (('reference', tb), ('fused', fused)):
        pred = predict(model, *inputs(tp, teams, ball, periods), periods)
        s = {c: ex2.score(truths[c], pred) for c in CONVENTIONS}
        row[tag] = float(np.mean([s[c]['spellF1'] for c in CONVENTIONS]))
        row[f'{tag}_macro'] = s['raw']['macroF1']
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', default=['N050', 'N030', 'N080'], choices=sorted(ARMS))
    args = ap.parse_args()
    summary = {}
    for arm in args.arms:
        rows = []
        (OUT / arm / 'rows').mkdir(parents=True, exist_ok=True)
        for m in dv.DFL:
            cached = OUT / arm / 'rows' / f'{m}.json'
            if cached.exists():
                rows.append(json.loads(cached.read_text()))
                continue
            row = run_match(m, ARMS[arm])
            rows.append(row)
            cached.write_text(json.dumps(row) + '\n')
            print(arm, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        (OUT / arm / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        d = [r['fused'] - r['reference'] for r in rows]
        summary[arm] = {'noise_m': ARMS[arm], 'deltas': d, 'mean_delta': float(np.mean(d)),
                        'wins': int(np.sum(np.array(d) > 0)),
                        'macro_mean_delta': float(np.mean([r['fused_macro'] - r['reference_macro'] for r in rows])),
                        'accepted_share': float(np.mean([r['accepted_share'] for r in rows])),
                        'missing_before': float(np.mean([r['missing_before'] for r in rows])),
                        'missing_after': float(np.mean([r['missing_after'] for r in rows]))}
        prev = json.loads((OUT / 'summary.json').read_text()) if (OUT / 'summary.json').exists() else {}
        prev.update(summary)
        (OUT / 'summary.json').write_text(json.dumps(prev, indent=1) + '\n')
        print(json.dumps(summary[arm], indent=1), flush=True)
