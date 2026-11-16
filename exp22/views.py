"""EXPERIMENT-22 views: the 90 EXPERIMENT-18 columns for Metrica views and for team-swapped training views.

    OMP_NUM_THREADS=1 .venv/bin/python -m exp22.views [--workers 4]     # resumable; external/exp22/views/
    OMP_NUM_THREADS=1 .venv/bin/python -m exp22.views --check           # rebuild one cached exp18 view, must match

A view is (ds, match, condition, purpose, swap). Unswapped SkillCorner and DFL views are read from the EXPERIMENT-18
cache (external/exp18/views/), PFF views from external/pff-cache/views/. Everything else is built here with the same
feature code (v3 features, v4 team shape, v6 control, exp18 geometry). swap = 1: x -> 1 - x, y -> 1 - y for players and
ball, team indices swapped, labels H <-> A; features recomputed from the mirrored tracking. Metrica eval views also
carry q_gbm: the frozen v6 model (the model inside v8) on their 72 columns, as pff.views does.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / 'external/exp22/views'
EXP18 = ROOT / 'external/exp18/views'
PFF = ROOT / 'external/pff-cache/views'
DFL_CONDITIONS = ('natural', 'gaps', 'contested')
_MODEL = None


def name(view):
    ds, m, c, p, s = view
    return f'{ds}-{m}-{c}-{p}' + ('-swap' if s else '')


def path_of(view):
    ds, m, c, p, s = view
    if ds == 'pff':
        return PFF / f'pff-{m}.npz'
    if ds in ('sc', 'dfl') and not s:
        return EXP18 / f'{name(view)}.npz'
    return CACHE / f'{name(view)}.npz'


def _v6():
    global _MODEL
    if _MODEL is None:
        from v6.infer import load_frozen
        _MODEL = load_frozen()
    return _MODEL


def arrays(ds, m, cond, purpose):
    """Tracking under the named condition: players, teams, ball, periods, labels."""
    from v2 import degrade as dg
    if ds == 'dfl':
        from v2 import idsse
        a = idsse.load(m)
    elif ds == 'sc':
        from v3 import skillcorner as sc
        a = sc.load(m)
    elif ds == 'met':
        from exp22 import metrica as me
        a = me.load(m)
    else:
        raise ValueError(ds)
    if ds == 'sc':
        players, ball = a['players'], a['ball']
    else:
        players, ball = dg.degrade(a, m, cond, purpose)
    return players, np.asarray(a['teams']), ball, np.asarray(a['periods']), np.asarray(a['labels'])


def mirror(players, teams, ball, labels):
    swap_lab = np.array(labels, dtype='<U1').copy()
    h, a = swap_lab == 'H', swap_lab == 'A'
    swap_lab[h], swap_lab[a] = 'A', 'H'
    return 1. - players, 1 - teams, 1. - ball, swap_lab


def features(players, teams, ball, periods):
    from v2 import methods as mt
    from v3 import features as f3
    from v4.shape import team_shape
    from v6.control import control_features
    from v6.infer import columns
    from exp18.geometry import geometry
    obs = mt.observe(players, teams, ball, periods)
    x72 = columns(f3.build(players, teams, ball, periods), team_shape(players, teams, periods)[0],
                  control_features(players, teams, ball, periods))
    x = np.hstack([x72, geometry(players, teams, ball, periods)])
    assert x.shape[1] == 90, x.shape
    return x72, x, obs


def build(view):
    from v2 import methods as mt
    ds, m, cond, purpose, swap = view
    players, teams, ball, periods, labels = arrays(ds, m, cond, purpose)
    if swap:
        players, teams, ball, labels = mirror(players, teams, ball, labels)
    x72, x, obs = features(players, teams, ball, periods)
    # DFL-style sources define visible from the finite ball speed (v4.develop.dfl_data); SkillCorner from the ball
    visible = obs['visible'] if ds == 'sc' else np.isfinite(x[:, 2])
    out = {'x': x.astype(np.float32), 'labels': labels, 'periods': periods, 'owner': obs['owner'],
           'visible': visible}
    if ds == 'met' and purpose == 'eval' and not swap:
        out['q_gbm'] = mt.learned_probs(_v6(), x72)
    return out


def load(view):
    path = path_of(view)
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if view[0] == 'pff' or (view[0] in ('sc', 'dfl') and not view[4]):
        raise SystemExit(f'missing upstream cache {path}')
    v = build(view)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **v)
    tmp.rename(path)
    return v


# ---------- view lists ----------

GAMES = ('1', '2', '3')                               # exp22.metrica.GAMES (kept here: no pandas/sklearn import)


def ids():
    """Match ids from the cache file names, as exp18.arm_a does, so the torch venv needs numpy only."""
    names = [p.name[:-4] for p in EXP18.glob('*.npz') if not p.name.endswith('.tmp.npz')]
    dfl = tuple(sorted({n.split('-')[1] for n in names if n.startswith('dfl-')}))
    sc = tuple(sorted({int(n.split('-')[1]) for n in names if n.startswith('sc-')}))
    if (len(dfl), len(sc)) != (7, 20):
        raise SystemExit(f'exp18 view cache incomplete: {len(dfl)} DFL, {len(sc)} SkillCorner')
    return {'dfl': dfl, 'sc': sc, 'met': GAMES}


def fold_of(ds, m):
    return sorted(ids()[ds]).index(m) % 5            # identical to v4.develop.fold_of


def train_views(fold, metrica=True):
    """fold None = all matches (final model). Each view enters unswapped and swapped."""
    i = ids()
    base = [('dfl', m, c, 'train') for m in i['dfl'] if fold is None or fold_of('dfl', m) != fold
            for c in DFL_CONDITIONS]
    if metrica:
        base += [('met', m, c, 'train') for m in i['met'] if fold is None or fold_of('met', m) != fold
                 for c in DFL_CONDITIONS]
    base += [('sc', m, 'real', 'eval') for m in i['sc'] if fold is None or fold_of('sc', m) != fold]
    return [v + (s,) for v in base for s in (0, 1)]


def eval_views(fold):
    i = ids()
    out = [('sc', m, 'real', 'eval', 0) for m in i['sc'] if fold_of('sc', m) == fold]
    out += [('dfl', m, c, 'eval', 0) for c in ('natural', 'contested') for m in i['dfl'] if fold_of('dfl', m) == fold]
    out += [('met', m, c, 'eval', 0) for c in ('natural', 'contested') for m in i['met'] if fold_of('met', m) == fold]
    return out


def pff_views():
    ms = sorted(int(p.name[4:-4]) for p in PFF.glob('pff-*.npz') if not p.name.endswith('.tmp.npz'))
    if len(ms) != 64:
        raise SystemExit(f'PFF view cache has {len(ms)} matches, expected 64')
    return [('pff', m, 'real', 'eval', 0) for m in ms]


def setting_of(view):
    return {('sc', 'real'): 'real', ('dfl', 'natural'): 'clean', ('dfl', 'contested'): 'gaps',
            ('met', 'natural'): 'met_clean', ('met', 'contested'): 'met_gaps', ('pff', 'real'): 'pff'}[(view[0], view[2])]


def _cache(view):
    load(view)
    return name(view)


def check():
    """The builder must reproduce a cached EXPERIMENT-18 view exactly (same code path, so a defect here shows)."""
    view = ('dfl', ids()['dfl'][0], 'gaps', 'train', 0)
    fresh = build(view)
    with np.load(path_of(view)) as z:
        cached = {k: z[k] for k in z.files}
    same = {k: bool(np.array_equal(fresh[k], cached[k], equal_nan=fresh[k].dtype.kind == 'f')) for k in cached}
    print(name(view), same)
    sw = build(view[:4] + (1,))
    print('swap: labels H<->A', bool(np.array_equal(sw['labels'] == 'H', cached['labels'] == 'A')),
          '| x differs', bool(not np.array_equal(sw['x'], cached['x'], equal_nan=True)))
    if not all(same.values()):
        raise SystemExit('builder does not reproduce the EXPERIMENT-18 cache')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        check()
        sys.exit(0)
    from concurrent.futures import ProcessPoolExecutor
    views = sorted(set(train_views(None) + [v for f in range(5) for v in eval_views(f)]))
    todo = [v for v in views if not path_of(v).exists()]
    # Metrica first: the fold models need them, and they are small
    todo.sort(key=lambda v: (v[0] != 'met', v[0], str(v[1]), v[2], v[4]))
    print(len(views), 'views,', len(todo), 'to build', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for n in pool.map(_cache, todo):
            print('cached', n, flush=True)
    print(len(views), 'views ready')
