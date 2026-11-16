"""v12 stage 2 (.venv-video, torch CPU): frozen network, 50/50 blend with v8's evidence, v8's decoder. LOCAL ONLY.

    OMP_NUM_THREADS=3 .venv-video/bin/python -m v12.predict --inputs <inputs.npz> --out <labels.npy>

Refuses unless the freeze (work/v12-evaluation/freeze.json) records the passed EXPERIMENT-22 gate, v12/final.pt and
every bound source match it, and torch/numpy equal requirements-v12.txt. Inference uses only the frozen weights and
the scaler stored with them; deterministic (fixed seed and threads, torch.use_deterministic_algorithms(True)).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp22.candidate import V8_SETTING, W_CONV  # noqa: E402
from v12 import env, freezecheck  # noqa: E402


def run(inputs):
    from exp22 import net
    from exp18 import decode as de
    d = np.load(inputs)
    model, scaler = net.load_model(freezecheck.MODEL)
    q_conv, c = net.predict(model, scaler, d['x'], d['periods'])
    q = W_CONV * q_conv.astype(float) + (1 - W_CONV) * d['q_v8'].astype(float)
    return de.decode(q, None, d['periods'], *V8_SETTING), q, c, d['periods']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite existing labels')
    env.check('torch')
    freezecheck.check()
    labels, q, c, periods = run(a.inputs)
    np.save(a.out, labels)
    np.savez(a.out.with_suffix('.probs.npz'), q=q.astype(np.float32), c=c)
    g = periods > 0
    print(json.dumps({'graded': int(g.sum()), 'd_share': round(float(np.mean(labels[g] == 'D')), 4)}))
