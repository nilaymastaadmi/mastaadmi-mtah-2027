"""Write work/v12-evaluation/freeze.json once (EXPERIMENT-22 gate passed, candidate adopted as v12).

    .venv-video/bin/python -m v12.freeze --inputs <inputs.npz from v12.inputs or exp22.candidate>

Binds: the model hash and its stored scaler digest, every bound source (v12/freezecheck.py BOUND), the package pins of
both requirement files, the blend weight and decoder, the digest of v8's labels on the match (base-path check) and
the expected CSV sha256. Re-frozen 16 Nov after v12/kickoff.py (second-half pre-kickoff frames set to D, organiser
FAQ): the previous freeze, expecting d0c865dc, is kept as work/v12-evaluation/freeze-d0c865dc.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp22.candidate import V8_SETTING, W_CONV  # noqa: E402
from v12 import env, freezecheck as fc  # noqa: E402
from v12.inputs import v8_labels_sha  # noqa: E402
from v12.kickoff import window  # noqa: E402

EXPECTED_CSV = '0cdc7351e1bcfeb345f4a4a3549c0b2b5d1abe62a2aaa381c10b9a9a93879d7a'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    a = p.parse_args()
    path = fc.FOLDER / 'freeze.json'
    if path.exists():
        raise SystemExit('freeze.json exists; refusing to overwrite')
    import torch
    gate = json.loads((ROOT / 'work/exp22/gate.json').read_text())
    blob = torch.load(fc.MODEL, weights_only=True)
    scaler = hashlib.sha256(b''.join(blob[k].numpy().tobytes() for k in ('med', 'iqr', 'nan_cols'))).hexdigest()
    d = np.load(a.inputs)
    freeze = {'version': 'v12', 'experiment': 'EXPERIMENT-22 recipe M', 'gate_verdict': gate['verdict'],
              'registration_sha256': gate['registration_sha256'], 'model': 'v12/final.pt',
              'model_sha256': fc.sha(fc.MODEL), 'scaler_sha256': scaler, 'c_in': int(blob['c_in']),
              'blend_weight_conv': W_CONV, 'decoder': list(V8_SETTING),
              'v8_labels_sha256': v8_labels_sha(d['q_v8'].astype(float), d['periods']),
              'kickoff_fix_d_frames_half_open': list(window()),
              'expected_csv_sha256': EXPECTED_CSV, 'pins': {'main': env.pins('main'), 'torch': env.pins('torch')},
              'training': {'torch_seed': 22, 'chunk_rng': [22, 5, 'epoch'], 'threads': 3, 'epochs': 12,
                           'data': 'SkillCorner 20 + DFL 7 + Metrica sample games 1-3, team-swapped'},
              'source_hashes': fc.bound_hashes()}
    fc.FOLDER.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=1) + '\n')
    print(json.dumps({k: v for k, v in freeze.items() if k not in ('source_hashes', 'pins')}, indent=1))
