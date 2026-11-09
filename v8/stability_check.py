"""Reproducibility checks for the fused v8 pipeline. Counts only; no labels.

    OMP_NUM_THREADS=2 .venv/bin/python -m v8.stability_check --data-dir data   # local Mac only
    OMP_NUM_THREADS=2 .venv/bin/python -m v8.stability_check --compare          # public DFL analogue

Alignment uses hard thresholds (residual <= 1.2 m), so a cross-platform last-bit difference could
in principle flip a frame's acceptance. This check (a) perturbs both views' coordinates by relative
1e-12..1e-8 and verifies the frame mapping and decoded labels are unchanged, and (b) reports the
smallest margin between any frame residual and the threshold.
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
from v3 import features as f3  # noqa: E402
from v4.shape import team_shape  # noqa: E402
from v6.control import control_features  # noqa: E402
from v6.infer import load_frozen as load_v6, predict  # noqa: E402
from v8 import align as al  # noqa: E402

SCALES = (1e-12, 1e-10, 1e-8)


def run(model, players, teams, ball, periods, bp, bb):
    res = al.align(players, bp, periods)
    fused, _ = al.fuse_ball(ball, bb, res)
    obs = mt.observe(players, teams, fused, periods)
    labels = predict(model, f3.build(players, teams, fused, periods), team_shape(players, teams, periods)[0],
                     control_features(players, teams, fused, periods), obs['owner'], periods)
    return res, labels


def check(model, players, teams, ball, periods, bp, bb):
    res, ref = run(model, players, teams, ball, periods, bp, bb)
    out = {'sha256': hashlib.sha256(''.join(ref).encode()).hexdigest()}
    r = res['residual'][np.isfinite(res['residual'])]
    out['min_margin_to_threshold_m'] = float(np.min(np.abs(r - al.CHAMFER_ACCEPT_M))) if len(r) else None
    for s in SCALES:
        rng = np.random.default_rng(0)
        p2 = players * (1 + s * rng.standard_normal(players.shape))
        b2 = bp * (1 + s * rng.standard_normal(bp.shape))
        res2, lab2 = run(model, p2, teams, ball, periods, b2, bb)
        out[f'mapping_changes {s:g}'] = int(np.sum(res2['mapping'] != res['mapping']))
        out[f'label_changes {s:g}'] = int(np.sum(lab2 != ref))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    model = load_v6()
    report = {}
    if args.data_dir:
        from possession import load_atd
        from v8.broadcast_audit import load_broadcast
        players, teams, ball, periods = load_atd(args.data_dir)
        bp, bb = load_broadcast(args.data_dir)
        report['competition'] = check(model, players, teams, ball, periods, bp, bb)
        report['competition'].pop('sha256')
    else:
        from v2 import degrade as dg, idsse
        from v8.simulate import second_view
        cache = ROOT / 'work/v8-evaluation/stability-cache'
        cache.mkdir(parents=True, exist_ok=True)
        for m in idsse.DEVELOPMENT[:2]:
            if (cache / f'{m}.json').exists():   # resumable: each match takes hours
                report[m] = json.loads((cache / f'{m}.json').read_text())
                print(m, report[m], '(cached)', flush=True)
                continue
            a = idsse.load(m)
            tp, tb = dg.degrade(a, m, 'contested', 'eval')
            seed = int(hashlib.sha256(f'{m}|analogue|20261104'.encode()).hexdigest()[:8], 16)
            bp, bb, _ = second_view(a['players'], a['ball'], a['periods'], delta=1234, seed=seed)
            report[m] = check(model, tp, a['teams'], tb, a['periods'], bp, bb)
            (cache / f'{m}.json').write_text(json.dumps(report[m]) + '\n')
            print(m, report[m], flush=True)
    ok = all(v == 0 for r in report.values() for k, v in r.items() if 'changes' in k)
    print('perturbation changes all zero:', ok)
    if args.compare:
        path = ROOT / 'work/v8-evaluation/reference-digests.json'
        digests = {k: r['sha256'] for k, r in report.items()}
        if path.exists():
            stored = json.loads(path.read_text())
            print('digests identical to', stored['platform'], ':', stored['digests'] == digests)
        else:
            import platform
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'platform': platform.platform(), 'digests': digests}, indent=1) + '\n')
            print('wrote reference digests')
    if not ok:
        raise SystemExit('perturbation changed alignment or labels; investigate before submission')
