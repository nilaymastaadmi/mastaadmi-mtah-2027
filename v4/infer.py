"""Frozen v4 inference (EXPERIMENT-06 selection). Competition data: LOCAL MACHINE ONLY.

    OMP_NUM_THREADS=2 .venv/bin/python -m v4.infer --data-dir data --out work/candidates/v4.csv

Refuses unless the adoption gate passed and the model, every bound source and the package versions
match the freeze. Writes the CSV, runs the official validator, prints the SHA-256. Never reads labels.
"""
import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import methods as mt  # noqa: E402
from v4 import develop as dv  # noqa: E402

FOLDER = ROOT / 'work/v4-evaluation'


def predict(model, x37, shape, owner, periods, variant, setting):
    q = mt.learned_probs(model, dv.columns(x37, shape, variant))
    return mt.blend(q, mt.prior50({'owner': owner}, periods), periods, *setting)


def check_freeze(freeze, selection, model_hash, sources, runtime):
    if not selection['gate']['passed'] or selection['variant'] != freeze['variant']:
        raise ValueError('adoption gate not passed or variant differs from selection')
    if list(selection['best'][freeze['variant']]['setting']) != list(freeze['setting']):
        raise ValueError('decoder setting differs from selection')
    if freeze['model_sha256'] != model_hash:
        raise ValueError('model differs from freeze')
    if freeze['source_hashes'] != sources:
        raise ValueError('bound source changed since freeze')
    for key in ('numpy', 'pandas', 'sklearn'):
        if freeze['runtime'][key] != runtime[key]:
            raise ValueError(f'{key} version differs from freeze')
    if freeze['runtime']['python'].rsplit('.', 1)[0] != runtime['python'].rsplit('.', 1)[0]:
        raise ValueError('Python minor version differs from freeze')
    return freeze['variant'], tuple(freeze['setting'])


def runtime():
    import numpy, pandas, sklearn
    return {'python': platform.python_version(), 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'sklearn': sklearn.__version__}


def load_frozen():
    from v4.freeze import bound_hashes
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    selection = json.loads((ROOT / 'work/v4-development/selection.json').read_text())
    raw = (FOLDER / 'model.pkl').read_bytes()
    variant, setting = check_freeze(freeze, selection, hashlib.sha256(raw).hexdigest(),
                                    bound_hashes(), runtime())
    return pickle.loads(raw), variant, setting  # project-trained, hash-verified above


def competition_inputs(data_dir):
    from possession import load_atd
    from v3 import features as f3
    from v4.shape import team_shape
    players, teams, ball, periods = load_atd(data_dir)
    obs = mt.observe(players, teams, ball, periods)
    return (f3.build(players, teams, ball, periods), team_shape(players, teams, periods)[0],
            obs['owner'], periods)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    try:
        model, variant, setting = load_frozen()
    except ValueError as error:
        raise SystemExit(f'v4 freeze check failed: {error}')
    x37, shape, owner, periods = competition_inputs(args.data_dir)
    labels = predict(model, x37, shape, owner, periods, variant, setting)
    from possession import write_submission
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.out), labels)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    print(f'v4 freeze checks and official validation passed; sha256 '
          f'{hashlib.sha256(args.out.read_bytes()).hexdigest()}. Labels were not read.')
