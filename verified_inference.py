"""Supported inference entry point: bind selection and holdout to the same model."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import sklearn

ROOT = Path(__file__).resolve().parent


def check_freeze(dev, held, upstream, actual_model_hash, actual_sources):
    if not held['gate']['provisional_gate_passed'] or dev['selected'] is None:
        raise ValueError('held-out gate has not passed')
    if held['selected'] != dev['selected']:
        raise ValueError('holdout settings differ from development selection')
    if not all(m['model_sha256'] == actual_model_hash for m in [dev, held, upstream]):
        raise ValueError('model identity differs across training, selection or holdout')
    if dev['source_hashes'] != actual_sources or held['source_hashes'] != actual_sources:
        raise ValueError('source differs from selection or holdout freeze')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    folder = ROOT / 'work/hybrid-evaluation'
    dev, held = [json.loads((folder / f'{name}.json').read_text()) for name in ['dev', 'holdout']]
    upstream = json.loads((ROOT / 'work/learned-evaluation/dev.json').read_text())
    model_hash = hashlib.sha256((ROOT / 'work/learned-evaluation/model.pkl').read_bytes()).hexdigest()
    actual_sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in dev['source_hashes']}
    check_freeze(dev, held, upstream, model_hash, actual_sources)
    provenance = json.loads((folder / 'provenance.json').read_text())
    runtime = {'python': platform.python_version(), 'numpy': numpy.__version__,
               'pandas': pandas.__version__, 'sklearn': sklearn.__version__}
    if any(provenance[k] != v for k, v in runtime.items()):
        raise SystemExit('Runtime versions differ from external evaluation')
    subprocess.run([sys.executable, str(ROOT / 'hybrid_possession.py'), 'infer',
                    '--data-dir', str(args.data_dir.resolve()), '--out', str(args.out.resolve())], check=True)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    print('Frozen model/settings/source/runtime checks and CSV validation passed.')
