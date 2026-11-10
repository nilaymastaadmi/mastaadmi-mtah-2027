"""v10 stage 1 (.venv): competition inputs for the EXPERIMENT-18 arm A model. LOCAL MACHINE ONLY, label-free.

    OMP_NUM_THREADS=1 .venv/bin/python -m v10.inputs --data-dir <data> --out work/v10-evaluation/inputs_run1.npz

Same tracking path as v8 (`v8.infer.fused_inputs`): tactical ATD, broadcast ATD aligned and fused into the missing
tactical ball frames, then the 72 v6 columns plus the exp18 geometry, cast to float32 exactly as the training caches
(`exp18.views`). Never reads ground truth or tags. Prints aggregates only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v3 import features as f3  # noqa: E402
from v4.shape import team_shape  # noqa: E402
from v6.control import control_features  # noqa: E402
from v6.infer import columns  # noqa: E402
from v8 import align as al  # noqa: E402
from exp18.geometry import geometry  # noqa: E402


def competition_x(data_dir):
    from possession import load_atd
    from v8.broadcast_audit import load_broadcast
    players, teams, ball, periods = load_atd(data_dir)
    bp, bb = load_broadcast(data_dir)
    res = al.align(players, bp, periods)
    fused, source = al.fuse_ball(ball, bb, res)
    x72 = columns(f3.build(players, teams, fused, periods), team_shape(players, teams, periods)[0],
                  control_features(players, teams, fused, periods))
    x = np.hstack([x72, geometry(players, teams, fused, periods)]).astype(np.float32)
    assert x.shape[1] == 90, x.shape   # 72 v6 columns + 18 geometry, as exp18.views
    return x, np.asarray(periods), source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite existing inputs')
    x, periods, source = competition_x(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=x, periods=periods)
    graded = periods > 0
    print(json.dumps({'frames': int(len(periods)), 'graded': int(graded.sum()), 'columns': int(x.shape[1]),
                      'recovered_ball_frames': int((source[graded] == 1).sum()),
                      'missing_ball_share': round(float((source[graded] == -1).mean()), 3)}))
