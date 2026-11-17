"""Cross-platform check of v13 on PUBLIC external matches only. No competition data, no PFF.

Same 17 public views as v10's and v12's checks (7 DFL natural, 7 DFL contested, 3 SkillCorner). Stage 1 (sklearn
venv) builds the four EXPERIMENT-27 views of each from the raw public tracking (`exp22.views.arrays`,
`exp27.views.transform`, `exp22.views.features`): the 90 network columns (float32) and v8's frozen v6 probabilities on
the 72 float64 columns, H and A swapped back for the team-swapped views. Stage 2 (torch venv) runs v12's frozen network
per view, averages both halves over the four views, blends 50/50 and decodes with v12's decoder, recording per-view
digests and arrays for a frame-by-frame comparison (`v10.portability.compare`).

    .venv/bin/python -m v13.portability inputs [--workers 4]
    .venv-video/bin/python -m v13.portability predict --out work/v13-evaluation/portability-mac.json \
        --arrays external/v13/portability-mac
    .venv-video/bin/python -m v13.portability compare A.json A_arrays B.json B_arrays [out.json]
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
from exp27.views import VIEWS as SYM, TEAM_SWAP, BACK, transform  # noqa: E402

INPUTS = ROOT / 'external/v13/portability-inputs'
_MODEL = []


def one(view):
    from exp22 import views as vw
    from v2 import methods as mt
    from v6.infer import load_frozen
    if not _MODEL:
        _MODEL.append(load_frozen())
    players, teams, ball, periods, _ = vw.arrays(*view)
    for k in SYM:
        path = INPUTS / f'{name(view)}-{k}.npz'
        if path.exists():
            continue
        p, t, b = transform(players, teams, ball, k)
        x72, x, _ = vw.features(p, t, b, periods)
        q = mt.learned_probs(_MODEL[0], x72)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, x=x.astype(np.float32), q_gbt=q[:, BACK] if TEAM_SWAP[k] else q, periods=np.asarray(periods))
        tmp.rename(path)
    return name(view)


def inputs(workers):
    from concurrent.futures import ProcessPoolExecutor
    INPUTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for n in pool.map(one, VIEWS):
            print('view ready', n, flush=True)


def predict(out, arrays):
    import torch
    from exp18 import decode as de
    from exp22 import net
    from v12 import env
    from v13 import freezecheck as fc
    versions = env.check('torch')
    freeze = json.loads((fc.FOLDER / 'freeze.json').read_text())
    if fc.sha(fc.MODEL) != freeze['model_sha256']:
        raise SystemExit('v12/final.pt differs from the v13 freeze')
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
    p.add_argument('--out', type=Path, default=ROOT / 'work/v13-evaluation/portability-mac.json')
    p.add_argument('--arrays', type=Path, default=ROOT / 'external/v13/portability-mac')
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
