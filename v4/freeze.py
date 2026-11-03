"""EXPERIMENT-06 freeze: retrain the selected variant on all 27 matches and record hashes and digests.

    OMP_NUM_THREADS=4 .venv/bin/python -m v4.freeze

Refuses unless the registered adoption gate passed. Writes work/v4-evaluation/{model.pkl,freeze.json,
reference-digests.json}. Reference digests use public DFL (natural, contested) and SkillCorner views.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import numpy
import pandas
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex2  # noqa: E402
from v4 import develop as dv  # noqa: E402

OUT = ROOT / 'work/v4-evaluation'
BOUND = ['v4/shape.py', 'v4/develop.py', 'v4/freeze.py', 'v4/infer.py', 'v3/features.py',
         'v3/skillcorner.py', 'work/EXPERIMENT-06.md', 'work/v4-development/selection.json']
REFERENCE_SC = 5  # first five SkillCorner ids


def bound_hashes():
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in BOUND}
    hashes.update(ex2.source_hashes())
    return hashes


def reference_views():
    for m in dv.DFL:
        for c in ('natural', 'contested'):
            yield f'dfl/{m}/{c}', dv.dfl_data(m, c, 'eval')
    for m in dv.SC[:REFERENCE_SC]:
        yield f'sc/{m}', dv.sc_data(m)


if __name__ == '__main__':
    from v4.infer import predict
    sel = json.loads((ROOT / 'work/v4-development/selection.json').read_text())
    if not sel['gate']['passed']:
        raise SystemExit('Adoption gate did not pass; nothing to freeze.')
    variant, setting = sel['variant'], sel['best'][sel['variant']]['setting']
    xs, ys = [], []
    for m in dv.DFL:
        for c in dg.CONDITIONS:
            d = dv.dfl_data(m, c, 'train')
            g = d['labels'] != 'X'
            xs.append(dv.columns(d['x'], d['shape'], variant)[g])
            ys.append(d['labels'][g])
    for m in dv.SC:
        d = dv.sc_data(m)
        g = (d['labels'] != 'X') & d['visible']
        xs.append(dv.columns(d['x'], d['shape'], variant)[g])
        ys.append(d['labels'][g])
    model = HistGradientBoostingClassifier(**ex2.HGB).fit(np.concatenate(xs), np.concatenate(ys))
    OUT.mkdir(parents=True, exist_ok=True)
    import pickle
    (OUT / 'model.pkl').write_bytes(pickle.dumps(model))
    model_hash = hashlib.sha256((OUT / 'model.pkl').read_bytes()).hexdigest()
    digests = {}
    for key, d in reference_views():
        labels = predict(model, d['x'], d['shape'], d['owner'], d['periods'], variant, setting)
        digests[key] = hashlib.sha256(''.join(labels).encode()).hexdigest()
    runtime = {'python': platform.python_version(), 'numpy': numpy.__version__,
               'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform()}
    freeze = {'variant': variant, 'setting': setting, 'model_sha256': model_hash,
              'training_frames': int(sum(len(y) for y in ys)), 'source_hashes': bound_hashes(),
              'runtime': runtime, 'registration': 'work/EXPERIMENT-06.md'}
    (OUT / 'freeze.json').write_text(json.dumps(freeze, indent=1) + '\n')
    (OUT / 'reference-digests.json').write_text(json.dumps({'platform': runtime['platform'],
                                                           'python': runtime['python'],
                                                           'digests': digests}, indent=1) + '\n')
    print(json.dumps({k: freeze[k] for k in ['variant', 'setting', 'model_sha256', 'training_frames']}))
