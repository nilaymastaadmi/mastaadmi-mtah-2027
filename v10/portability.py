"""Cross-platform check of v10 on PUBLIC external matches only (Codex review 10 Nov, P2). No competition data.

Runs the frozen v10 model (v10/final.pt) and decoder setting (work/v10-evaluation/freeze.json) on 7 DFL natural,
7 DFL contested and 3 SkillCorner views, built from the raw public data by `exp18.views`, and records per-view
digests of the inputs, of q rounded to 1e-6 and of the decoded labels. Arrays are saved too, so two platforms can
be compared frame by frame.

    .venv/bin/python -m v10.portability inputs                                   # sklearn env: build view caches
    .venv-video/bin/python -m v10.portability predict --out work/v10-evaluation/portability-mac.json \
        --arrays external/exp18/portability-mac                                  # torch env
    .venv-video/bin/python -m v10.portability compare A.json A_arrays B.json B_arrays
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DFL = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')
SC = (1874553, 1886347, 1899585)
VIEWS = ([('dfl', m, 'natural', 'eval') for m in DFL] + [('dfl', m, 'contested', 'eval') for m in DFL]
         + [('sc', m, 'real', 'eval') for m in SC])


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def name(v):
    return '-'.join(str(p) for p in v)


def inputs():
    from exp18 import views as vw
    for v in VIEWS:
        vw.load(v)
        print('view ready', name(v), flush=True)


def predict(out, arrays):
    import torch
    from exp18 import arm_a as aa, decode as de
    freeze = json.loads((ROOT / 'work/v10-evaluation/freeze.json').read_text())
    model_path = ROOT / 'v10/final.pt'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != freeze['model_sha256']:
        raise SystemExit('v10/final.pt differs from the freeze')
    model, scaler = aa.load_model(model_path)
    arrays.mkdir(parents=True, exist_ok=True)
    rows = {}
    for v in VIEWS:
        d = aa.load(v)
        q, c = aa.predict(model, scaler, d['x'], d['periods'])
        labels = de.decode(q.astype(float), c.astype(float), d['periods'], *freeze['setting'])
        rows[name(v)] = {'frames': int(len(labels)), 'x_sha256': sha(d['x']), 'q_c_sha256': sha(np.r_[q.ravel(), c]),
                         'q_round6_sha256': sha(np.round(q.astype(np.float64), 6)), 'labels_sha256': sha(labels.astype('<U1'))}
        np.savez_compressed(arrays / f'{name(v)}.npz', q=q, c=c, labels=labels.astype('<U1'),
                            x_sha=np.array(rows[name(v)]['x_sha256']))
        print(name(v), rows[name(v)]['labels_sha256'][:12], flush=True)
    meta = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'blas': torch.backends.cpu.get_cpu_capability(),
            'config': [l.strip() for l in torch.__config__.show().splitlines() if 'BLAS' in l or 'MKL' in l][:6],
            'setting': freeze['setting'], 'model_sha256': freeze['model_sha256']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'meta': meta, 'views': rows}, indent=1) + '\n')


def compare(ja, da, jb, db):
    a, b = json.loads(Path(ja).read_text()), json.loads(Path(jb).read_text())
    res = {'views': 0, 'input_differs': 0, 'q_round6_differs': 0, 'label_views_differ': 0, 'label_frames_differ': 0,
           'frames': 0, 'max_abs_q_diff': 0., 'per_view': {}}
    for n in a['views']:
        za, zb = np.load(Path(da) / f'{n}.npz'), np.load(Path(db) / f'{n}.npz')
        dq = float(np.max(np.abs(za['q'].astype(np.float64) - zb['q'].astype(np.float64))))
        nd = int(np.sum(za['labels'] != zb['labels']))
        res['views'] += 1
        res['frames'] += a['views'][n]['frames']
        res['input_differs'] += a['views'][n]['x_sha256'] != b['views'][n]['x_sha256']
        res['q_round6_differs'] += a['views'][n]['q_round6_sha256'] != b['views'][n]['q_round6_sha256']
        res['label_views_differ'] += nd > 0
        res['label_frames_differ'] += nd
        res['max_abs_q_diff'] = max(res['max_abs_q_diff'], dq)
        res['per_view'][n] = {'max_abs_q_diff': dq, 'label_frames_differ': nd}
    res['meta'] = {'a': a['meta'], 'b': b['meta']}
    print(json.dumps({k: v for k, v in res.items() if k != 'per_view'}, indent=1))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inputs', 'predict', 'compare'])
    p.add_argument('--out', type=Path, default=ROOT / 'work/v10-evaluation/portability-mac.json')
    p.add_argument('--arrays', type=Path, default=ROOT / 'external/exp18/portability-mac')
    p.add_argument('paths', nargs='*')
    args = p.parse_args()
    if args.stage == 'inputs':
        inputs()
    elif args.stage == 'predict':
        predict(args.out, args.arrays)
    else:
        compare(*args.paths)
