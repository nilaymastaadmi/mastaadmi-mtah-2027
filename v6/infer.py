"""Frozen v6 inference (EXPERIMENT-10: v4 features + ball control). Competition data: LOCAL MACHINE ONLY.

    OMP_NUM_THREADS=2 .venv/bin/python -m v6.infer --data-dir data --out work/candidates/v6.csv
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
from v6.control import control_features  # noqa: E402

FOLDER = ROOT / 'work/v6-evaluation'
SETTING = (1., .5)


def columns(x37, shape, ctrl):
    return np.hstack([dv.columns(x37, shape, 'C'), ctrl])


def predict(model, x37, shape, ctrl, owner, periods):
    q = mt.learned_probs(model, columns(x37, shape, ctrl))
    return mt.blend(q, mt.prior50({'owner': owner}, periods), periods, *SETTING)


def runtime():
    import numpy, pandas, sklearn
    return {'python': platform.python_version(), 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'sklearn': sklearn.__version__}


def check_freeze(freeze, selection, model_hash, sources, rt):
    if not selection['passed']:
        raise ValueError('EXPERIMENT-10 gate did not pass')
    if freeze['model_sha256'] != model_hash:
        raise ValueError('model differs from freeze')
    if freeze['source_hashes'] != sources:
        raise ValueError('bound source changed since freeze')
    for key in ('numpy', 'pandas', 'sklearn'):
        if freeze['runtime'][key] != rt[key]:
            raise ValueError(f'{key} version differs from freeze')
    if freeze['runtime']['python'].rsplit('.', 1)[0] != rt['python'].rsplit('.', 1)[0]:
        raise ValueError('Python minor version differs from freeze')


def load_frozen():
    from v6.freeze import bound_hashes
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    selection = json.loads((ROOT / 'work/v6-development/selection.json').read_text())
    raw = (FOLDER / 'model.pkl').read_bytes()
    check_freeze(freeze, selection, hashlib.sha256(raw).hexdigest(), bound_hashes(), runtime())
    return pickle.loads(raw)  # project-trained, hash-verified above


def competition_inputs(data_dir):
    from possession import load_atd
    from v3 import features as f3
    from v4.shape import team_shape
    players, teams, ball, periods = load_atd(data_dir)
    obs = mt.observe(players, teams, ball, periods)
    return (f3.build(players, teams, ball, periods), team_shape(players, teams, periods)[0],
            control_features(players, teams, ball, periods), obs['owner'], periods)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    try:
        model = load_frozen()
    except ValueError as error:
        raise SystemExit(f'v6 freeze check failed: {error}')
    labels = predict(model, *competition_inputs(args.data_dir))
    from possession import write_submission
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.out), labels)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    print(f'v6 freeze checks and official validation passed; sha256 '
          f'{hashlib.sha256(args.out.read_bytes()).hexdigest()}. Labels were not read.')
