"""Cross-platform reproducibility checks for frozen v4. Reads no labels; prints counts and digests.

Perturbs, by relative noise far above last-bit platform differences, (a) the learned probabilities
(libm exp/log) and (b) every feature whose value depends on floating-point summation order: mean and
per-team player speeds and their rolling means, and team-shape means/standard deviations. All other
features come from exact IEEE operations. Then compares label digests on public reference views.

    OMP_NUM_THREADS=2 .venv/bin/python -m v4.stability_check --compare      # public data needed
    OMP_NUM_THREADS=2 .venv/bin/python -m v4.stability_check --data-dir data   # local Mac only
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
from v4 import develop as dv  # noqa: E402
from v4.infer import load_frozen, predict  # noqa: E402

SCALES = (1e-12, 1e-10, 1e-8)
SUM_X37 = [7, 14, 15, 24, 29, 34]  # mean speed, per-team speeds, rolling means of mean speed
SUM_SHAPE_KEYS = ('_cx', '_sx', '_sy', '_adv')


def shape_sum_columns():
    from v4.shape import team_shape
    toy = np.random.default_rng(0).uniform(.2, .8, (60, 4, 2))
    _, names = team_shape(toy, np.array([0, 0, 1, 1]), np.ones(60, int))
    return [i for i, n in enumerate(names) if n.endswith(SUM_SHAPE_KEYS)]


def check(model, variant, setting, x37, shape, owner, periods, seeds=3):
    reference = predict(model, x37, shape, owner, periods, variant, setting)
    out = {'sha256': hashlib.sha256(''.join(reference).encode()).hexdigest()}
    cols = dv.columns(x37, shape, variant)
    q = mt.learned_probs(model, cols)
    prior = mt.prior50({'owner': owner}, periods)
    shape_cols = shape_sum_columns()
    for scale in SCALES:
        p_changed, f_changed = 0, 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            p_changed = max(p_changed, int(np.sum(mt.blend(q * (1 + scale * rng.standard_normal(q.shape)),
                                                           prior, periods, *setting) != reference)))
            x2, s2 = x37.copy(), shape.copy()
            x2[:, SUM_X37] *= 1 + scale * rng.standard_normal((len(x2), len(SUM_X37)))
            s2[:, shape_cols] *= 1 + scale * rng.standard_normal((len(s2), len(shape_cols)))
            f_changed = max(f_changed, int(np.sum(predict(model, x2, s2, owner, periods, variant, setting)
                                                  != reference)))
        out[f'probabilities {scale:g}'] = p_changed
        out[f'summed_features {scale:g}'] = f_changed
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    model, variant, setting = load_frozen()
    report = {}
    if args.data_dir:
        from v4.infer import competition_inputs
        report['competition'] = check(model, variant, setting, *competition_inputs(args.data_dir))
        report['competition'].pop('sha256')  # counts only
    else:
        from v4.freeze import reference_views
        for key, d in reference_views():
            report[key] = check(model, variant, setting, d['x'], d['shape'], d['owner'], d['periods'])
            print(key, {k: v for k, v in report[key].items() if k != 'sha256'}, flush=True)
    ok = all(v == 0 for r in report.values() for k, v in r.items() if k != 'sha256')
    print('perturbation flips all zero:', ok)
    if args.compare and not args.data_dir:
        stored = json.loads((ROOT / 'work/v4-evaluation/reference-digests.json').read_text())
        same = stored['digests'] == {k: r['sha256'] for k, r in report.items()}
        print(f"digests identical to {stored['platform']} / Python {stored['python']}: {same}")
        if not same:
            raise SystemExit('platform output differs from reference; do not submit before resolving')
    if not ok:
        raise SystemExit('perturbation changed decoded labels; investigate before submission')
