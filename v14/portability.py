"""Cross-platform check of v14 on PUBLIC external matches only. No competition data, no PFF.

Same 17 public views (7 DFL natural, 7 DFL contested, 3 SkillCorner) and the same stage 1 as v13's check
(`v13.portability inputs`: the four EXPERIMENT-27 views of each, built from the raw public tracking; inputs do not
depend on the network). Stage 2 (torch venv) runs v14's network (`v14/final.pt`, EXPERIMENT-29 candidate B) per view,
averages both halves over the four views, blends 50/50 and decodes with v12's decoder, recording per-view digests and
arrays for a frame-by-frame comparison (`v10.portability.compare`).

    .venv/bin/python -m v14.portability inputs [--workers 4]
    .venv-video/bin/python -m v14.portability predict --out work/v14-evaluation/portability-mac.json \
        --arrays external/v14/portability-mac
    .venv-video/bin/python -m v14.portability compare A.json A_arrays B.json B_arrays [out.json]
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
from v13.portability import INPUTS, inputs  # noqa: E402
from exp27.views import VIEWS as SYM, TEAM_SWAP, BACK  # noqa: E402


def predict(out, arrays):
    import torch
    from exp18 import decode as de
    from exp22 import net
    from v12 import env
    from v14 import freezecheck as fc
    versions = env.check('torch')
    freeze = json.loads((fc.FOLDER / 'freeze.json').read_text())
    if fc.sha(fc.MODEL) != freeze['model_sha256']:
        raise SystemExit('v14/final.pt differs from the v14 freeze')
    model, scaler = net.load_model(fc.MODEL)
    arrays.mkdir(parents=True, exist_ok=True)
    rows = {}
    for v in VIEWS:
        conv, gbt, xs = {}, {}, hashlib.sha256()
        for k in SYM:
            with np.load(INPUTS / f'{name(v)}-{k}.npz') as z:
                q, _ = net.predict(model, scaler, z['x'], z['periods'])
                conv[k] = (q[:, BACK] if TEAM_SWAP[k] else q).astype(float)
                gbt[k] = z['q_gbt'].astype(float)
                periods = z['periods']
                xs.update(np.ascontiguousarray(z['x']).tobytes())
        q = .5 * (sum(conv.values()) / 4) + .5 * (sum(gbt.values()) / 4)
        labels = de.decode(q, None, periods, .5, 0.)
        rows[name(v)] = {'frames': int(len(labels)), 'x_sha256': xs.hexdigest(), 'q_round6_sha256': sha(np.round(q, 6)),
                         'labels_sha256': sha(labels.astype('<U1'))}
        np.savez_compressed(arrays / f'{name(v)}.npz', q=q, labels=labels.astype('<U1'),
                            x_sha=np.array(rows[name(v)]['x_sha256']))
        print(name(v), rows[name(v)]['labels_sha256'][:12], flush=True)
    meta = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'versions': versions,
            'config': [l.strip() for l in torch.__config__.show().splitlines() if 'BLAS' in l or 'MKL' in l][:6],
            'model_sha256': freeze['model_sha256'], 'v6_model_sha256': freeze['v6_model_sha256']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'meta': meta, 'views': rows}, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inputs', 'predict', 'compare'])
    p.add_argument('--out', type=Path, default=ROOT / 'work/v14-evaluation/portability-mac.json')
    p.add_argument('--arrays', type=Path, default=ROOT / 'external/v14/portability-mac')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('paths', nargs='*')
    args = p.parse_args()
    if args.stage == 'inputs':
        inputs(args.workers)
    elif args.stage == 'predict':
        predict(args.out, args.arrays)
    else:
        res = compare(*args.paths[:4])
        if len(args.paths) > 4:
            Path(args.paths[4]).write_text(json.dumps(res, indent=1) + '\n')
