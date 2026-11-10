"""EXPERIMENT-18 shared inputs: views, per-frame features (v6 columns + geometry), labels and change targets.

    OMP_NUM_THREADS=4 .venv/bin/python -m exp18.views          # builds every cache (resumable)

Cached as float32 under ignored external/exp18/views/. Folds and training views follow the v8 recipe
(`v8.analogue.fold_model`): DFL outside the fold under the three train-purpose conditions, SkillCorner outside the
fold. Held-out views: SkillCorner real, DFL natural (clean) and DFL contested (gaps), eval purpose.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, idsse  # noqa: E402
from v3 import skillcorner as sc  # noqa: E402
from v4 import develop as dv  # noqa: E402
from v6 import develop as d6  # noqa: E402
from exp18.geometry import NAMES as GEO_NAMES, geometry  # noqa: E402
from exp18.targets import change_events, near  # noqa: E402,F401  (re-exported)

CACHE = ROOT / 'external/exp18/views'
N_V6 = 72
CONTROL = slice(60, 72)          # the 12 v6 control columns (v6.control.control_features) inside the 72
SETTINGS = {'real': ('sc', 'real'), 'clean': ('dfl', 'natural'), 'gaps': ('dfl', 'contested')}


def fold_of(ds, match):
    return dv.fold_of(dv.SC if ds == 'sc' else dv.DFL, match)


def train_views(fold):
    out = [('dfl', m, c, 'train') for m in dv.DFL if dv.fold_of(dv.DFL, m) != fold for c in dg.CONDITIONS]
    return out + [('sc', m, 'real', 'eval') for m in dv.SC if dv.fold_of(dv.SC, m) != fold]


def all_train_views():
    return [('dfl', m, c, 'train') for m in dv.DFL for c in dg.CONDITIONS] + [('sc', m, 'real', 'eval') for m in dv.SC]


def eval_views(fold=None):
    out = []
    for setting, (ds, cond) in SETTINGS.items():
        for m in (dv.SC if ds == 'sc' else dv.DFL):
            if fold is None or fold_of(ds, m) == fold:
                out.append((ds, m, cond, 'eval'))
    return out


def setting_of(view):
    return {('sc', 'real'): 'real', ('dfl', 'natural'): 'clean', ('dfl', 'contested'): 'gaps'}[(view[0], view[2])]


def name(view):
    ds, m, c, p = view
    return f'{ds}-{m}-{c}-{p}'


def _build(view):
    ds, m, cond, purpose = view
    if ds == 'dfl':
        d = dv.dfl_data(m, cond, purpose)
        a = idsse.load(m)
        players, ball = dg.degrade(a, m, cond, purpose)
        teams = a['teams']
    else:
        d = dv.sc_data(m)
        a = sc.load(m)
        players, ball, teams = a['players'], a['ball'], a['teams']
    x = np.hstack([d6.features(ds, m, cond, purpose, d), geometry(players, teams, ball, d['periods'])])
    assert x.shape[1] == N_V6 + len(GEO_NAMES)
    return {'x': x.astype(np.float32), 'labels': d['labels'], 'periods': d['periods'], 'owner': d['owner'],
            'visible': d['visible']}


def load(view):
    path = CACHE / f'{name(view)}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    v = _build(view)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **v)
    tmp.rename(path)
    return v


def _cache(view):
    load(view)
    return name(view)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    views = sorted(set(all_train_views() + eval_views()))
    todo = [v for v in views if not (CACHE / f'{name(v)}.npz').exists()]
    with ProcessPoolExecutor(4) as pool:
        for n in pool.map(_cache, todo):
            print('cached', n, flush=True)
    print(len(views), 'views ready')
