"""Frozen v6 model with broadcast ball fusion (EXPERIMENT-13). Competition data: LOCAL MACHINE ONLY.

    OMP_NUM_THREADS=2 .venv/bin/python -m v8.infer --data-dir data --out work/candidates/v8.csv

Loads the frozen, hash-verified v6 model unchanged; aligns the broadcast ATD to the tactical
timeline; fills missing tactical ball frames from accepted broadcast frames; then computes v6
features and decodes exactly as v6. Refuses unless work/v8-analogue/selection.json records a passed
gate and the bound alignment sources match work/v8-evaluation/freeze.json.
"""
import argparse
import hashlib
import json
import subprocess
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

BOUND = ['v8/align.py', 'v8/infer.py', 'v8/broadcast_audit.py', 'work/EXPERIMENT-13.md']


def bound_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in BOUND}


def check_freeze():
    sel = json.loads((ROOT / 'work/v8-analogue/selection.json').read_text())
    if not sel['passed']:
        raise ValueError('EXPERIMENT-13 analogue gate did not pass')
    freeze = json.loads((ROOT / 'work/v8-evaluation/freeze.json').read_text())
    if freeze['source_hashes'] != bound_hashes():
        raise ValueError('alignment sources changed since freeze')
    return freeze


def fused_inputs(data_dir):
    from possession import load_atd
    from v8.broadcast_audit import load_broadcast
    players, teams, ball, periods = load_atd(data_dir)
    bp, bb = load_broadcast(data_dir)
    res = al.align(players, bp, periods)
    fused, source = al.fuse_ball(ball, bb, res)
    obs = mt.observe(players, teams, fused, periods)
    return (f3.build(players, teams, fused, periods), team_shape(players, teams, periods)[0],
            control_features(players, teams, fused, periods), obs['owner'], periods), source, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    try:
        check_freeze()
        model = load_v6()
    except ValueError as error:
        raise SystemExit(f'v8 freeze check failed: {error}')
    inputs, source, res = fused_inputs(args.data_dir)
    labels = predict(model, *inputs)
    from possession import write_submission
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.out), labels)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    graded = inputs[4] > 0
    print(json.dumps({'recovered_ball_frames': int((source[graded] == 1).sum()),
                      'missing_after_share': round(float((source[graded] == -1).mean()), 3),
                      'offsets': res['offsets'], 'flip': list(res['flip']),
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))
    print('v8 freeze checks and official validation passed. Labels were not read.')
