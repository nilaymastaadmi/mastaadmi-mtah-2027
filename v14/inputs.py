"""v14 stage 1 (.venv): competition inputs for the four symmetric views. LOCAL MACHINE ONLY, label-free.

    OMP_NUM_THREADS=1 .venv/bin/python -m v14.inputs --data-dir <data> --out <inputs.npz>

v12's tracking path (tactical ATD, broadcast ATD aligned and fused into missing tactical ball frames, `v8.infer`
freeze checked) with the warm-up fix: the second half's features start at the organiser kickoff (video frame 113,742,
`v12.kickoff`), so frames 110,808 to 113,741 never enter a feature window. Then, for each EXPERIMENT-27 view (identity;
touchline mirror; full swap with teams swapped; goal-line mirror with teams swapped), the 90 network columns (float32)
and v8's frozen v6 probabilities on the 72 float64 columns, H and A swapped back for the team-swapped views. This is
exactly `exp27.match inputs --warmup`, run after the v14 freeze and package checks. Never reads ground truth, tags or
evaluation files.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v12 import env  # noqa: E402
from v14 import freezecheck  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite existing inputs')
    env.check('main')
    freezecheck.check()
    from exp27 import match
    match.inputs(argparse.Namespace(data_dir=a.data_dir, out=a.out, warmup=True))
