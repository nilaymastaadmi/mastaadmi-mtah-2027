"""v12 stage 1 (.venv): competition inputs and v8's evidence. LOCAL MACHINE ONLY, label-free.

    OMP_NUM_THREADS=1 .venv/bin/python -m v12.inputs --data-dir <data> --out <inputs.npz>

v8's tracking path, run once: tactical ATD, broadcast ATD aligned and fused into missing tactical ball frames
(`v8.infer` freeze checked). From the fused tracking: the 90 network columns (72 v6 columns + 18 exp18 geometry,
float32, as the training caches) and q_v8, v8's frozen v6 model on the float64 72 columns (as `v8.infer`). Checks
that v8's decoder on q_v8 reproduces the frozen v8 label digest, so the blend partner is exactly v8. Never reads
ground truth, tags or evaluation files. Prints aggregates only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp22.candidate import V8_SETTING  # noqa: E402
from v12 import env, freezecheck  # noqa: E402


def compute(data_dir):
    from possession import load_atd
    from v2 import methods as mt
    from v3 import features as f3
    from v4.shape import team_shape
    from v6.control import control_features
    from v6.infer import columns, load_frozen
    from v8 import align as al
    from v8.broadcast_audit import load_broadcast
    from v8.infer import check_freeze
    from exp18.geometry import geometry
    check_freeze()
    players, teams, ball, periods = load_atd(data_dir)
    bp, bb = load_broadcast(data_dir)
    fused, source = al.fuse_ball(ball, bb, al.align(players, bp, periods))
    x72 = columns(f3.build(players, teams, fused, periods), team_shape(players, teams, periods)[0],
                  control_features(players, teams, fused, periods))
    q_v8 = mt.learned_probs(load_frozen(), x72)
    x = np.hstack([x72, geometry(players, teams, fused, periods)]).astype(np.float32)
    assert x.shape[1] == 90, x.shape
    return x, np.asarray(periods), q_v8, source


def v8_labels_sha(q_v8, periods):
    from exp18 import decode as de
    labels = de.decode(q_v8, None, periods, *V8_SETTING)
    return hashlib.sha256(np.ascontiguousarray(labels.astype('<U1')).tobytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite existing inputs')
    versions = env.check('main')
    freeze = freezecheck.check()
    x, periods, q_v8, source = compute(a.data_dir)
    v8sha = v8_labels_sha(q_v8, periods)
    if v8sha != freeze['v8_labels_sha256']:
        raise SystemExit('v8 base path does not reproduce the frozen v8 labels')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, x=x, periods=periods, q_v8=q_v8)
    g = periods > 0
    print(json.dumps({'frames': int(len(periods)), 'graded': int(g.sum()), 'v8_base_reproduced': True,
                      'missing_ball_share': round(float((source[g] == -1).mean()), 3), 'versions_ok': True}))
