"""Supported v2 inference entry (EXPERIMENT-04 candidate). LOCAL MACHINE ONLY for competition data.

    OMP_NUM_THREADS=2 .venv/bin/python -m v2.verified_inference --data-dir data --out work/candidates/v2.csv

Refuses unless the one-time holdout passed its promotion gate, the model and every bound source
match the development and holdout freezes, and the package versions match training. Never print
restricted arrays, frames or labels into a hosted model context. The competition labels are not read.
"""
import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'work/v2-evaluation'


def check_v2_freeze(dev, held, provenance, model_hash, sources, runtime):
    if held['candidate'] != dev['candidate'] or held['candidate'] != 'blend_learned':
        raise ValueError('frozen candidate is not the registered blend_learned selection')
    if not held['promotion_gate']['passed']:
        raise ValueError('holdout promotion gate did not pass')
    if not (dev['model_sha256'] == held['model_sha256'] == model_hash):
        raise ValueError('model identity differs from development/holdout freeze')
    if dev['source_hashes'] != sources or held['source_hashes'] != sources:
        raise ValueError('bound source changed since freeze')
    for key in ['sklearn', 'numpy', 'pandas']:
        if provenance['runtime'][key] != runtime[key]:
            raise ValueError(f'{key} version differs from training runtime')
    if provenance['runtime']['python'].rsplit('.', 1)[0] != runtime['python'].rsplit('.', 1)[0]:
        raise ValueError('Python minor version differs from training runtime')
    return tuple(dev['selected']['blend_learned'])


def runtime():
    return {'python': platform.python_version(), 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'sklearn': sklearn.__version__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    from v2.experiment import source_hashes
    dev, held, provenance = [json.loads((FOLDER / f'{n}.json').read_text())
                             for n in ['dev', 'holdout', 'provenance']]
    model_path = FOLDER / 'model.pkl'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    try:
        weight, penalty = check_v2_freeze(dev, held, provenance, model_hash, source_hashes(), runtime())
    except ValueError as error:
        raise SystemExit(f'v2 freeze check failed: {error}')
    with model_path.open('rb') as f:
        model = pickle.load(f)  # locally trained, hash-verified above; never load external pickles
    from possession import load_atd, write_submission
    from v2 import methods as mt
    players, teams, ball, periods = load_atd(args.data_dir)
    obs = mt.observe(players, teams, ball, periods)
    q = mt.learned_probs(model, mt.learned_inputs(players, teams, ball, periods))
    labels = mt.blend(q, mt.prior50(obs, periods), periods, weight, penalty)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.out), labels)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f'v2 freeze checks and official validation passed; sha256 {digest}. Labels were not read.')
