"""v10 stage 3 (.venv): labels -> submission CSV, then the official validator. LOCAL MACHINE ONLY.

    .venv/bin/python -m v10.write --labels work/v10-evaluation/labels_run1.npy --out work/v10-evaluation/v10_run1.csv
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing candidate')
    from possession import write_submission
    labels = np.load(args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(args.out), labels)
    subprocess.run([sys.executable, str(ROOT / 'starter-repo/evaluation/validate_submission.py'),
                    str(args.out.resolve())], check=True, stdout=subprocess.DEVNULL)
    print(f'official validation passed; sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}. '
          'Labels were not read.')
