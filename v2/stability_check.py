"""Cross-platform reproducibility checks for frozen v2. Reads no labels; prints counts and digests.

1. Perturbation: relative noise far above last-bit platform differences (about 1e-16) is applied to
   (a) learned probabilities (libm exp/log differences) and (b) mean player speed, the only feature
   whose value depends on floating-point summation order. Other features are produced by exact IEEE
   operations (subtraction, multiplication, division, min, correctly rounded sqrt) and are
   bit-identical across IEEE-754 platforms. Many of them sit exactly on tree thresholds by design
   (constants such as 0.5 for missing ball), so perturbing them is not a meaningful platform test.
2. Digests: sha256 of v2 labels on public DFL matches (natural and contested). Generate on one
   machine with --write, compare on another with --compare. Identical digests are direct evidence
   that the platform reproduces the frozen pipeline.

    OMP_NUM_THREADS=2 .venv/bin/python -m v2.stability_check --compare   # needs external/idsse
    OMP_NUM_THREADS=2 .venv/bin/python -m v2.stability_check --data-dir data   # local Mac only
"""
import argparse
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import methods as mt  # noqa: E402

SCALES = (1e-12, 1e-10, 1e-8)
MEAN_SPEED = 7  # column of mean player speed in v1 learned_possession.inputs
DIGESTS = ROOT / 'work/v2-evaluation/reference-digests.json'


def check(model, players, teams, ball, periods, weight, penalty, seeds=3):
    obs = mt.observe(players, teams, ball, periods)
    x = mt.learned_inputs(players, teams, ball, periods)
    q = mt.learned_probs(model, x)
    prior = mt.prior50(obs, periods)
    reference = mt.blend(q, prior, periods, weight, penalty)
    out = {'sha256': hashlib.sha256(''.join(reference).encode()).hexdigest()}
    for scale in SCALES:
        prob_changed, speed_changed = 0, 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            noisy_q = q * (1 + scale * rng.standard_normal(q.shape))
            prob_changed = max(prob_changed, int(np.sum(
                mt.blend(noisy_q, prior, periods, weight, penalty) != reference)))
            noisy_x = x.copy()
            noisy_x[:, MEAN_SPEED] *= 1 + scale * rng.standard_normal(len(x))
            speed_changed = max(speed_changed, int(np.sum(mt.blend(
                mt.learned_probs(model, noisy_x), prior, periods, weight, penalty) != reference)))
        out[f'probabilities {scale:g}'] = prob_changed
        out[f'mean_speed {scale:g}'] = speed_changed
    return out


def frozen():
    folder = ROOT / 'work/v2-evaluation'
    dev = json.loads((folder / 'dev.json').read_text())
    raw = (folder / 'model.pkl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != dev['model_sha256']:
        raise SystemExit('frozen v2 model hash mismatch')
    return pickle.loads(raw), dev['selected']['blend_learned']  # project-trained, hash-verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='competition data (local Mac only)')
    parser.add_argument('--write', action='store_true', help='store public reference digests')
    parser.add_argument('--compare', action='store_true', help='compare with stored digests')
    args = parser.parse_args()
    model, (weight, penalty) = frozen()
    report = {}
    if args.data_dir:
        from possession import load_atd
        report['competition'] = check(model, *load_atd(args.data_dir), weight, penalty)
        report['competition'].pop('sha256')  # never print a digest of competition output here
    else:
        from v2 import idsse, degrade as dg
        for match in idsse.DEVELOPMENT + idsse.HOLDOUT:
            a = idsse.load(match)
            for condition in ('natural', 'contested'):
                players, ball = dg.degrade(a, match, condition, 'eval')
                report[f'{match}/{condition}'] = check(model, players, a['teams'], ball,
                                                       a['periods'], weight, penalty)
                print(match, condition, report[f'{match}/{condition}'], flush=True)
    perturbation_ok = all(v == 0 for r in report.values() for k, v in r.items() if k != 'sha256')
    print('perturbation flips all zero:', perturbation_ok)
    digests = {k: r['sha256'] for k, r in report.items() if 'sha256' in r}
    if args.write and digests:
        DIGESTS.write_text(json.dumps({'platform': platform.platform(),
                                       'python': platform.python_version(),
                                       'digests': digests}, indent=1) + '\n')
        print(f'wrote {DIGESTS.relative_to(ROOT)}')
    if args.compare and digests:
        stored = json.loads(DIGESTS.read_text())
        same = stored['digests'] == digests
        print(f"digests identical to {stored['platform']} / Python {stored['python']}: {same}")
        if not same:
            raise SystemExit('platform output differs from reference; do not submit before resolving')
    if not perturbation_ok:
        raise SystemExit('perturbation changed decoded labels; investigate before submission')
