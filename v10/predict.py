"""v10 stage 2 (.venv-video, torch CPU): frozen arm A model + change-aware decoder -> labels. LOCAL MACHINE ONLY.

    .venv-video/bin/python -m v10.predict --inputs work/v10-evaluation/inputs_run1.npz --out work/v10-evaluation/labels_run1.npy

Refuses unless work/v10-evaluation/freeze.json records a passed EXPERIMENT-18 gate for arm A and the model and
bound sources match it. Deterministic: fixed seed and thread count, `torch.use_deterministic_algorithms(True)`.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp18 import arm_a as aa, decode as de  # noqa: E402

FOLDER = ROOT / 'work/v10-evaluation'
MODEL = ROOT / 'v10/final.pt'   # tracked copy of external/exp18/arm_a/models/final.pt
# Every repo file on the inference path (Codex review 10 Nov): a reordered feature column fails the freeze.
BOUND = ['exp18/arm_a.py', 'exp18/decode.py', 'exp18/geometry.py', 'exp18/targets.py', 'exp18/views.py',
         'v10/inputs.py', 'v10/predict.py', 'v10/write.py', 'v10/freeze.py', 'v10/run.sh', 'requirements-v10.txt',
         'work/EXPERIMENT-18.md', 'work/exp18/gate-A.json', 'work/exp18/result-A.json',
         'possession.py', 'learned_possession.py', 'external_validation.py', 'starter-repo/games.py',
         'starter-repo/starter/baseline.py', 'starter-repo/evaluation/possession_metrics.py',
         'starter-repo/evaluation/validate_submission.py',
         'v2/methods.py', 'v2/degrade.py', 'v2/experiment.py', 'v2/idsse.py',
         'v3/features.py', 'v3/decode.py', 'v3/develop.py', 'v3/skillcorner.py',
         'v4/develop.py', 'v4/shape.py', 'v6/control.py', 'v6/infer.py', 'v8/align.py', 'v8/broadcast_audit.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_hashes():
    return {p: sha(ROOT / p) for p in BOUND}


def check_freeze():
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    gate = json.loads((ROOT / 'work/exp18/gate-A.json').read_text())
    if freeze.get('arm') != 'A' or not gate['passed']:
        raise ValueError('EXPERIMENT-18 arm A gate did not pass')
    if freeze['model_sha256'] != sha(MODEL):
        raise ValueError('model differs from freeze')
    if freeze['source_hashes'] != bound_hashes():
        raise ValueError('bound source changed since freeze')
    return freeze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite existing labels')
    try:
        freeze = check_freeze()
    except ValueError as error:
        raise SystemExit(f'v10 freeze check failed: {error}')
    model, scaler = aa.load_model(MODEL)
    with np.load(args.inputs) as z:
        x, periods = z['x'], z['periods']
    q, c = aa.predict(model, scaler, x, periods)
    labels = de.decode(q.astype(float), c.astype(float), periods, *freeze['setting'])
    np.save(args.out, labels)
    graded = periods > 0
    print(json.dumps({'q_c_sha256': aa.digest(q, c), 'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
                      'possessions_changes': int(np.sum(labels[1:][graded[1:]] != labels[:-1][graded[1:]])),
                      'D_share': round(float(np.mean(labels[graded] == 'D')), 3)}))
