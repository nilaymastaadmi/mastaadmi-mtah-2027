"""Write work/v14-evaluation/freeze.json once (EXPERIMENT-29 passed checks 1 to 4, candidate B packaged as v14).

    .venv-video/bin/python -m v14.freeze

Binds: v14's network `v14/final.pt` (hash, equal to the EXPERIMENT-29 build's final-B.pt, and the stored scaler
digest), v8's frozen v6 model hash, every bound source (v14/freezecheck.py BOUND), the package pins of both
requirement files, the views, blend weight and decoder, the warm-up and pre-kickoff windows and the expected CSV
sha256 (the EXPERIMENT-29 build, work/exp29-evaluation/freeze.json). PFF FC's terms (EXPERIMENT-29 check 5) are
checked outside this script; nothing here records them.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v12 import env  # noqa: E402
from v12.kickoff import window  # noqa: E402
from v14 import freezecheck as fc  # noqa: E402

EXPECTED_CSV = 'ace6e2f701aed67ca0715d79f7ccc4f571ee599983e112a6d8d01aca4336cd30'

if __name__ == '__main__':
    path = fc.FOLDER / 'freeze.json'
    if path.exists():
        raise SystemExit('freeze.json exists; refusing to overwrite')
    import torch
    gate = json.loads((ROOT / 'work/exp29/gate.json').read_text())
    build = json.loads((ROOT / 'work/exp29-evaluation/freeze.json').read_text())
    passed = (gate['verdict'].startswith('PASS') and gate['selected'] == 'B' and build['final_verdict'] == 'PASS'
              and build['candidate'] == 'B' and build['csv_sha256'] == EXPECTED_CSV
              and build['model_sha256'] == fc.sha(fc.MODEL))
    blob = torch.load(fc.MODEL, weights_only=True)
    scaler = hashlib.sha256(b''.join(blob[k].numpy().tobytes() for k in ('med', 'iqr', 'nan_cols'))).hexdigest()
    freeze = {'version': 'v14', 'experiment': 'EXPERIMENT-29 candidate B (v13 with the network retrained on '
                                              'SkillCorner 20, DFL 7, Metrica 3 and PFF 64, PFF weight 1.0)',
              'gate_verdict': 'PASS' if passed else 'NOT PASSED', 'registration_sha256': gate['registration_sha256'],
              'model': 'v14/final.pt', 'model_sha256': fc.sha(fc.MODEL), 'scaler_sha256': scaler,
              'c_in': int(blob['c_in']), 'v6_model': 'work/v6-evaluation/model.pkl', 'v6_model_sha256': fc.sha(fc.V6_MODEL),
              'trained_on_pff': True,
              'views': ['identity', 'touchline mirror (y -> 1 - y)', 'full swap (x, y mirrored, teams swapped)',
                        'goal-line mirror with teams swapped (x -> 1 - x)'],
              'average': 'mean over the four views of each half (H/A mapped back), then 0.5 conv + 0.5 boosted',
              'decoder': [.5, 0.], 'warmup_periods_zeroed_half_open': list(window()),
              'kickoff_fix_d_frames_half_open': list(window()), 'expected_csv_sha256': EXPECTED_CSV,
              'pins': {'main': env.pins('main'), 'torch': env.pins('torch')}, 'source_hashes': fc.bound_hashes()}
    fc.FOLDER.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=1) + '\n')
    print(json.dumps({k: v for k, v in freeze.items() if k not in ('source_hashes', 'pins')}, indent=1))
