"""Cross-platform reproducibility checks for frozen v6 (counts and public digests only; no labels).

    OMP_NUM_THREADS=2 .venv/bin/python -m v6.stability_check --compare      # public data needed
    OMP_NUM_THREADS=2 .venv/bin/python -m v6.stability_check --data-dir data   # local Mac only

Perturbs probabilities and every summation-order-dependent input (v4 speed/shape columns). Control
features use exact differences, minima and integer cumulative sums, so they are platform-exact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import methods as mt  # noqa: E402
from v4.stability_check import SCALES, SUM_X37, shape_sum_columns  # noqa: E402
from v6.infer import columns, load_frozen, predict, SETTING  # noqa: E402


def check(model, x37, shape, ctrl, owner, periods, seeds=3):
    ref = predict(model, x37, shape, ctrl, owner, periods)
    out = {'sha256': hashlib.sha256(''.join(ref).encode()).hexdigest()}
    q = mt.learned_probs(model, columns(x37, shape, ctrl))
    prior = mt.prior50({'owner': owner}, periods)
    sc = shape_sum_columns()
    for s in SCALES:
        pc = fc = 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            pc = max(pc, int(np.sum(mt.blend(q * (1 + s * rng.standard_normal(q.shape)), prior, periods, *SETTING) != ref)))
            x2, s2 = x37.copy(), shape.copy()
            x2[:, SUM_X37] *= 1 + s * rng.standard_normal((len(x2), len(SUM_X37)))
            s2[:, sc] *= 1 + s * rng.standard_normal((len(s2), len(sc)))
            fc = max(fc, int(np.sum(predict(model, x2, s2, ctrl, owner, periods) != ref)))
        out[f'probabilities {s:g}'], out[f'summed_features {s:g}'] = pc, fc
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--compare', action='store_true')
    a = p.parse_args()
    model = load_frozen()
    report = {}
    if a.data_dir:
        from v6.infer import competition_inputs
        report['competition'] = check(model, *competition_inputs(a.data_dir))
        report['competition'].pop('sha256')
    else:
        from v6.freeze import reference_views
        for k, d, c in reference_views():
            report[k] = check(model, d['x'], d['shape'], c, d['owner'], d['periods'])
    ok = all(v == 0 for r in report.values() for k, v in r.items() if k != 'sha256')
    print('perturbation flips all zero:', ok)
    if a.compare and not a.data_dir:
        stored = json.loads((ROOT / 'work/v6-evaluation/reference-digests.json').read_text())
        same = stored['digests'] == {k: r['sha256'] for k, r in report.items()}
        print(f"digests identical to {stored['platform']} / Python {stored['python']}: {same}")
        if not same:
            raise SystemExit('platform output differs from reference; do not submit before resolving')
    if not ok:
        raise SystemExit('perturbation changed decoded labels; investigate before submission')
