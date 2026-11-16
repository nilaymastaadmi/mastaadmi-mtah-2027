"""Cross-platform check of v12 on PUBLIC external matches only. No competition data, no PFF.

Same 17 public views as v10's check (7 DFL natural, 7 DFL contested, 3 SkillCorner), built from the raw public data
by `exp18.views`. Stage 1 (sklearn venv) adds v8's evidence: the frozen v6 model on each view's 72 columns. Stage 2
(torch venv) runs the frozen v12 network, the 50/50 blend and v8's decoder, and records per-view digests of the
inputs, of q rounded to 1e-6 and of the labels, with arrays for a frame-by-frame comparison (`v10.portability.compare`).

    .venv/bin/python -m v12.portability inputs
    .venv-video/bin/python -m v12.portability predict --out work/v12-evaluation/portability-mac.json \
        --arrays external/v12/portability-mac
    .venv-video/bin/python -m v12.portability compare A.json A_arrays B.json B_arrays
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
from v10.portability import VIEWS, compare, name, sha  # noqa: E402

GBM = ROOT / 'external/v12/portability-gbm'


def inputs():
    from exp18 import views as vw
    from v2 import methods as mt
    from v6.infer import load_frozen
    model = load_frozen()
    GBM.mkdir(parents=True, exist_ok=True)
    for v in VIEWS:
        d = vw.load(v)
        q = mt.learned_probs(model, d['x'][:, :72].astype(np.float64))
        np.savez(GBM / f'{name(v)}.npz', q=q)
        print('view ready', name(v), flush=True)


def predict(out, arrays):
    import torch
    from exp18 import arm_a as aa, decode as de
    from exp22 import net
    from exp22.candidate import V8_SETTING, W_CONV
    from v12 import env, freezecheck as fc
    versions = env.check('torch')
    freeze = json.loads((fc.FOLDER / 'freeze.json').read_text())
    if fc.sha(fc.MODEL) != freeze['model_sha256']:
        raise SystemExit('v12/final.pt differs from the freeze')
    model, scaler = net.load_model(fc.MODEL)
    arrays.mkdir(parents=True, exist_ok=True)
    rows = {}
    for v in VIEWS:
        d = aa.load(v)
        q_conv, c = net.predict(model, scaler, d['x'], d['periods'])
        q = W_CONV * q_conv.astype(float) + (1 - W_CONV) * np.load(GBM / f'{name(v)}.npz')['q']
        labels = de.decode(q, None, d['periods'], *V8_SETTING)
        rows[name(v)] = {'frames': int(len(labels)), 'x_sha256': sha(d['x']), 'q_conv_c_sha256': sha(np.r_[q_conv.ravel(), c]),
                         'q_round6_sha256': sha(np.round(q, 6)), 'labels_sha256': sha(labels.astype('<U1'))}
        np.savez_compressed(arrays / f'{name(v)}.npz', q=q, c=c, labels=labels.astype('<U1'),
                            x_sha=np.array(rows[name(v)]['x_sha256']))
        print(name(v), rows[name(v)]['labels_sha256'][:12], flush=True)
    meta = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'versions': versions,
            'config': [l.strip() for l in torch.__config__.show().splitlines() if 'BLAS' in l or 'MKL' in l][:6],
            'model_sha256': freeze['model_sha256']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'meta': meta, 'views': rows}, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inputs', 'predict', 'compare'])
    p.add_argument('--out', type=Path, default=ROOT / 'work/v12-evaluation/portability-mac.json')
    p.add_argument('--arrays', type=Path, default=ROOT / 'external/v12/portability-mac')
    p.add_argument('paths', nargs='*')
    args = p.parse_args()
    if args.stage == 'inputs':
        inputs()
    elif args.stage == 'predict':
        predict(args.out, args.arrays)
    else:
        res = compare(*args.paths[:4])
        if len(args.paths) > 4:
            Path(args.paths[4]).write_text(json.dumps(res, indent=1) + '\n')
