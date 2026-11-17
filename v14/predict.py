"""v14 stage 2 (.venv-video, torch CPU): four-view averaging with the PFF-trained network, v12's decoder. LOCAL ONLY.

    OMP_NUM_THREADS=3 .venv-video/bin/python -m v14.predict --inputs <inputs.npz> --out <labels.npy>

Exactly v13 (EXPERIMENT-27 T2) with one change: the conv half's network is `v14/final.pt`, recipe M of EXPERIMENT-22
retrained on SkillCorner 20 + DFL 7 + Metrica 3 + PFF 64 (EXPERIMENT-29 candidate B, PFF weight 1.0; its own scaler).
For each view the network gives q_conv (H and A swapped back for the team-swapped views); q = 0.5 x mean over the four
views of q_conv + 0.5 x mean over the four views of v8's boosted q, decoded with v12's decoder over the warm-up
periods, then the pre-kickoff fix (`v12.kickoff`: frames 110,808 to 113,741 are D). Runs `exp27.match predict --arm
T2` with the network and its hash taken from the v14 freeze. Refuses unless the v14 freeze, every bound source and
the torch/numpy pins match. Deterministic (fixed seed and threads, torch.use_deterministic_algorithms(True)).
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
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite existing labels')
    env.check('torch')
    freeze = freezecheck.check()
    from exp27 import match
    match.MODEL, match.MODEL_SHA = freezecheck.MODEL, freeze['model_sha256']
    match.predict(argparse.Namespace(inputs=a.inputs, out=a.out, arm='T2'))
