"""v13 stage 2 (.venv-video, torch CPU): four-view averaging of v12's frozen models, v12's decoder. LOCAL ONLY.

    OMP_NUM_THREADS=3 .venv-video/bin/python -m v13.predict --inputs <inputs.npz> --out <labels.npy>

For each view, v12's frozen network (`v12/final.pt`, sha256 e68c5767) gives q_conv; H and A are swapped back for the
team-swapped views. q = 0.5 x mean over the four views of q_conv + 0.5 x mean over the four views of v8's boosted q
(EXPERIMENT-27 candidate T2), decoded with v12's decoder over the warm-up periods, then the pre-kickoff fix
(`v12.kickoff`: frames 110,808 to 113,741 are D). This is exactly `exp27.match predict --arm T2` with the network read
from `v12/final.pt` (the same file as EXPERIMENT-22's final model; the hash is checked twice). Refuses unless the v13
freeze, every bound source and the torch/numpy pins match. Deterministic (fixed seed and threads,
torch.use_deterministic_algorithms(True)).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v12 import env  # noqa: E402
from v13 import freezecheck  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite existing labels')
    env.check('torch')
    freezecheck.check()
    from exp27 import match
    match.MODEL = freezecheck.MODEL
    match.predict(argparse.Namespace(inputs=a.inputs, out=a.out, arm='T2'))
