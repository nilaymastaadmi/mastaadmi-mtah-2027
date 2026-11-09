"""EXPERIMENT-15: short-gap ball interpolation, out of fold on SkillCorner real views and DFL contested views.

    OMP_NUM_THREADS=2 .venv/bin/python -m v8.gap_study

Fold models come from v8/analogue.cached_model (EXPERIMENT-10 recipe, cached in external/models/).
Resumable: per-view rows cached in work/v8-gaps/rows/. Writes work/v8-gaps/{rows.jsonl,selection.json}.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2, idsse  # noqa: E402
from v3 import skillcorner as sc  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v5.labels import relabel  # noqa: E402
from v6.infer import predict  # noqa: E402
from v8.analogue import CONVENTIONS, cached_model, inputs  # noqa: E402
from v8.interp import fill_gaps  # noqa: E402

OUT = ROOT / 'work/v8-gaps'
ARMS = (0, 12, 25, 50, 125)


def views():
    for m in dv.SC:
        a = sc.load(m)
        yield 'sc', m, dv.fold_of(dv.SC, m), a['players'], a['teams'], a['ball'], a['periods'], a['labels']
    for m in dv.DFL:
        a = idsse.load(m)
        tp, tb = dg.degrade(a, m, 'contested', 'eval')
        yield 'dfl', m, dv.fold_of(dv.DFL, m), tp, a['teams'], tb, a['periods'], a['labels']


if __name__ == '__main__':
    (OUT / 'rows').mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, m, fold, players, teams, ball, periods, labels in views():
        cached = OUT / 'rows' / f'{ds}-{m}.json'
        if cached.exists():
            rows.append(json.loads(cached.read_text()))
            continue
        model = cached_model(fold)
        truths = {c: relabel(labels, periods, c) for c in CONVENTIONS}
        row = {'view': f'{ds}-{m}', 'missing_before': float(np.mean(~np.isfinite(ball[periods > 0]).all(axis=1)))}
        for k in ARMS:
            filled, source = fill_gaps(ball, periods, k)
            pred = predict(model, *inputs(players, teams, filled, periods), periods)
            s = {c: ex2.score(truths[c], pred) for c in CONVENTIONS}
            row[f'K{k}'] = float(np.mean([s[c]['spellF1'] for c in CONVENTIONS]))
            row[f'K{k}_macro'] = s['raw']['macroF1']
            row[f'K{k}_missing'] = float(np.mean(source[periods > 0] == -1))
        rows.append(row)
        cached.write_text(json.dumps(row) + '\n')
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    scv = [r for r in rows if r['view'].startswith('sc-')]
    dflv = [r for r in rows if r['view'].startswith('dfl-')]
    means = {k: float(np.mean([r[f'K{k}'] for r in scv])) for k in ARMS}
    best = max(means.values())
    sel = min(k for k in ARMS if means[k] >= best - .005)
    d = [r[f'K{sel}'] - r['K0'] for r in scv]
    con = [r[f'K{sel}'] - r['K0'] for r in dflv]
    macro = float(np.mean([r[f'K{sel}_macro'] - r['K0_macro'] for r in scv]))
    checks = {'sc_mean_ge_.015': float(np.mean(d)) >= .015, 'sc_wins_ge_14of20': int(np.sum(np.array(d) > 0)) >= 14,
              'dfl_contested_ge_-.005': float(np.mean(con)) >= -.005, 'sc_macro_ge_-.02': macro >= -.02}
    report = {'selected_K': sel, 'sc_means': means, 'sc_delta': float(np.mean(d)), 'sc_wins': int(np.sum(np.array(d) > 0)),
              'dfl_contested_delta': float(np.mean(con)), 'sc_macro_delta': macro,
              'dfl_contested_means': {k: float(np.mean([r[f'K{k}'] for r in dflv])) for k in ARMS},
              'checks': checks, 'passed': all(checks.values()) and sel > 0}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
