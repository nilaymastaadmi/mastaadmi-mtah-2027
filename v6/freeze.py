"""EXPERIMENT-10 freeze: retrain variant D on all 27 matches; record hashes and public reference digests.

    OMP_NUM_THREADS=4 .venv/bin/python -m v6.freeze
"""
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v6 import develop as d6  # noqa: E402

OUT = ROOT / 'work/v6-evaluation'
BOUND = ['v6/control.py', 'v6/develop.py', 'v6/freeze.py', 'v6/infer.py', 'v4/shape.py', 'v4/develop.py',
         'v3/features.py', 'v3/skillcorner.py', 'work/EXPERIMENT-10.md', 'work/v6-development/selection.json']


def bound_hashes():
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in BOUND}
    hashes.update(ex2.source_hashes())
    return hashes


def reference_views():
    for m in dv.DFL:
        for c in ('natural', 'contested'):
            d = dv.dfl_data(m, c, 'eval')
            yield f'dfl/{m}/{c}', d, d6.ctrl('dfl', m, c, 'eval')
    for m in dv.SC[:5]:
        d = dv.sc_data(m)
        yield f'sc/{m}', d, d6.ctrl('sc', m, 'real', 'eval')


if __name__ == '__main__':
    from v6.infer import predict
    sel = json.loads((ROOT / 'work/v6-development/selection.json').read_text())
    if not sel['passed']:
        raise SystemExit('EXPERIMENT-10 gate did not pass; nothing to freeze.')
    xs, ys = [], []
    for m in dv.DFL:
        for c in dg.CONDITIONS:
            d = dv.dfl_data(m, c, 'train')
            g = d['labels'] != 'X'
            xs.append(d6.features('dfl', m, c, 'train', d)[g]); ys.append(d['labels'][g])
    for m in dv.SC:
        d = dv.sc_data(m)
        g = (d['labels'] != 'X') & d['visible']
        xs.append(d6.features('sc', m, 'real', 'eval', d)[g]); ys.append(d['labels'][g])
    model = HistGradientBoostingClassifier(**ex2.HGB).fit(np.concatenate(xs), np.concatenate(ys))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'model.pkl').write_bytes(pickle.dumps(model))
    model_hash = hashlib.sha256((OUT / 'model.pkl').read_bytes()).hexdigest()
    digests = {k: hashlib.sha256(''.join(predict(model, d['x'], d['shape'], c, d['owner'], d['periods'])).encode()).hexdigest()
               for k, d, c in reference_views()}
    import numpy, pandas, sklearn
    rt = {'python': platform.python_version(), 'numpy': numpy.__version__, 'pandas': pandas.__version__,
          'sklearn': sklearn.__version__, 'platform': platform.platform()}
    freeze = {'variant': 'D', 'setting': [1., .5], 'model_sha256': model_hash,
              'training_frames': int(sum(len(y) for y in ys)), 'source_hashes': bound_hashes(), 'runtime': rt,
              'registration': 'work/EXPERIMENT-10.md'}
    (OUT / 'freeze.json').write_text(json.dumps(freeze, indent=1) + '\n')
    (OUT / 'reference-digests.json').write_text(json.dumps({'platform': rt['platform'], 'python': rt['python'],
                                                           'digests': digests}, indent=1) + '\n')
    print(json.dumps({k: freeze[k] for k in ['variant', 'model_sha256', 'training_frames']}))
