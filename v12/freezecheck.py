"""Shared v12 freeze check (standard library only): bound source hashes, model hash and expected digests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'work/v12-evaluation'
MODEL = ROOT / 'v12/final.pt'
# Every repo file on the v12 inference path, plus the evidence it rests on.
BOUND = ['v12/env.py', 'v12/freezecheck.py', 'v12/inputs.py', 'v12/predict.py', 'v12/freeze.py', 'v12/run.sh',
         'requirements.txt', 'requirements-v12.txt', 'exp22/net.py', 'exp22/views.py', 'exp22/candidate.py',
         'exp18/arm_a.py', 'exp18/targets.py', 'exp18/decode.py', 'exp18/geometry.py', 'v10/write.py',
         'possession.py', 'learned_possession.py', 'starter-repo/games.py',
         'starter-repo/evaluation/possession_metrics.py', 'starter-repo/evaluation/validate_submission.py',
         'v2/methods.py', 'v3/features.py', 'v4/develop.py', 'v4/shape.py', 'v6/control.py', 'v6/infer.py',
         'v8/align.py', 'v8/broadcast_audit.py', 'v8/infer.py', 'work/EXPERIMENT-22.md', 'work/exp22/gate.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_hashes():
    return {p: sha(ROOT / p) for p in BOUND}


def check():
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    if freeze.get('gate_verdict') != 'PASS':
        raise SystemExit('freeze does not record a passed EXPERIMENT-22 gate')
    if freeze['model_sha256'] != sha(MODEL):
        raise SystemExit('v12/final.pt differs from the freeze')
    now = bound_hashes()
    changed = sorted(p for p in set(now) | set(freeze['source_hashes']) if now.get(p) != freeze['source_hashes'].get(p))
    if changed:
        raise SystemExit(f'bound sources changed since the freeze: {changed}')
    return freeze
