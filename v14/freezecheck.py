"""Shared v14 freeze check (standard library only): bound source hashes, model hashes and the passed gate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'work/v14-evaluation'
MODEL = ROOT / 'v14/final.pt'                      # EXPERIMENT-29 candidate B (recipe M plus PFF, weight 1.0)
V6_MODEL = ROOT / 'work/v6-evaluation/model.pkl'   # v8's frozen boosted model (also checked by v6.infer.load_frozen)
# Every repo file on the v14 inference path, plus the evidence it rests on.
BOUND = ['v14/freezecheck.py', 'v14/inputs.py', 'v14/predict.py', 'v14/freeze.py', 'v14/run.sh',
         'exp27/views.py', 'exp27/match.py', 'v12/env.py', 'v12/freezecheck.py', 'v12/inputs.py', 'v12/kickoff.py',
         'starter-repo/starter/load_smart_tagging.py', 'requirements.txt', 'requirements-v12.txt',
         'exp22/net.py', 'exp22/views.py', 'exp22/candidate.py', 'exp18/arm_a.py', 'exp18/targets.py',
         'exp18/decode.py', 'exp18/geometry.py', 'v10/write.py', 'possession.py', 'learned_possession.py',
         'starter-repo/games.py', 'starter-repo/evaluation/possession_metrics.py',
         'starter-repo/evaluation/validate_submission.py', 'v2/methods.py', 'v3/features.py', 'v4/develop.py',
         'v4/shape.py', 'v6/control.py', 'v6/infer.py', 'v8/align.py', 'v8/broadcast_audit.py', 'v8/infer.py',
         'work/EXPERIMENT-27.md', 'work/exp27/gate.json', 'work/EXPERIMENT-29.md', 'work/exp29/gate.json',
         'work/exp29-evaluation/freeze.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_hashes():
    return {p: sha(ROOT / p) for p in BOUND}


def check():
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    if freeze.get('gate_verdict') != 'PASS':
        raise SystemExit('freeze does not record a passed EXPERIMENT-29 bar')
    if freeze['model_sha256'] != sha(MODEL):
        raise SystemExit('v14/final.pt differs from the v14 freeze')
    if freeze['v6_model_sha256'] != sha(V6_MODEL):
        raise SystemExit('the frozen v6 model differs from the v14 freeze')
    now = bound_hashes()
    changed = sorted(p for p in set(now) | set(freeze['source_hashes']) if now.get(p) != freeze['source_hashes'].get(p))
    if changed:
        raise SystemExit(f'bound sources changed since the freeze: {changed}')
    return freeze
