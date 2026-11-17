"""EXPERIMENT-27 views: four symmetric views per evaluation view, features recomputed, boosted q (work/EXPERIMENT-27.md).

    OMP_NUM_THREADS=1 .venv/bin/python -B -m exp27.views [--workers 7]    # external/exp27/views/<view>-<V>.npz
    OMP_NUM_THREADS=1 .venv/bin/python -B -m exp27.views --check           # H3

V0 identity; V1 y -> 1 - y (teams unchanged); V2 full swap (exp22.views.mirror); V3 x -> 1 - x with teams swapped.
Features from the transformed tracking with exp22.views.features (72 v6 columns float64 for the boosted half, 90
columns float32 for the conv half). Boosted q: v8's v6-recipe fold model (SkillCorner, DFL) or the frozen v6 model
(Metrica, PFF), with H and A swapped back for V2 and V3. Saves x (float32, for exp27.conv), periods and q_gbt.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp22 import views as vw  # noqa: E402

CACHE = ROOT / 'external/exp27/views'
VIEWS = ('V0', 'V1', 'V2', 'V3')
TEAM_SWAP = {'V0': False, 'V1': False, 'V2': True, 'V3': True}
BACK = [1, 0, 2]                                   # H <-> A (state order H, A, D)
_MODELS = {}


def transform(players, teams, ball, k):
    teams = np.asarray(teams)
    if k == 'V0':
        return players, teams, ball
    if k == 'V2':
        p, t, b, _ = vw.mirror(players, teams, ball, np.full(len(ball), 'X'))
        return p, t, b
    p, b = players.copy(), ball.copy()
    axis = 1 if k == 'V1' else 0
    p[..., axis] = 1. - p[..., axis]
    b[:, axis] = 1. - b[:, axis]
    return p, (teams if k == 'V1' else 1 - teams), b


def tracking(view):
    ds, m, cond, purpose, _ = view
    if ds == 'pff':
        from pff import load as pl
        a = pl.load(m)
        return a['players'], np.asarray(a['teams']), a['ball'], np.asarray(a['periods'])
    players, teams, ball, periods, _ = vw.arrays(ds, m, cond, purpose)
    return players, teams, ball, periods


def boosted_model(view):
    ds = view[0]
    key = f'fold{vw.fold_of(ds, view[1])}' if ds in ('sc', 'dfl') else 'frozen'
    if key not in _MODELS:
        if key == 'frozen':
            from v6.infer import load_frozen
            _MODELS[key] = load_frozen()
        else:
            _MODELS[key] = pickle.loads((ROOT / f'external/models/v6-recipe-{key}.pkl').read_bytes())  # project-trained
    return _MODELS[key]


def out_name(view):
    return f'pff-{view[1]}' if view[0] == 'pff' else vw.name(view)


def path_of(view, k):
    return CACHE / f'{out_name(view)}-{k}.npz'


def build(view, k, tr=None):
    from v2 import methods as mt
    players, teams, ball, periods = tr if tr is not None else tracking(view)
    p, t, b = transform(players, teams, ball, k)
    x72, x, _ = vw.features(p, t, b, periods)
    q = mt.learned_probs(boosted_model(view), x72)
    if TEAM_SWAP[k]:
        q = q[:, BACK]
    return {'x': x.astype(np.float32), 'periods': np.asarray(periods), 'q_gbt': q}


def _cache(view):
    todo = [k for k in VIEWS if not path_of(view, k).exists()]
    if todo:
        tr = tracking(view)
        for k in todo:
            d = build(view, k, tr)
            tmp = path_of(view, k).with_suffix('.tmp.npz')
            np.savez(tmp, **d)
            tmp.rename(path_of(view, k))
    return out_name(view)


def all_views():
    return [v for f in range(5) for v in vw.eval_views(f)] + vw.pff_views()


def check():
    """H3: V2 recomputed on one DFL training view equals the cached EXPERIMENT-22 swap view's 90 columns."""
    view = ('dfl', vw.ids()['dfl'][0], 'gaps', 'train', 0)
    players, teams, ball, periods, _ = vw.arrays(*view[:4])
    p, t, b = transform(players, teams, ball, 'V2')
    x = vw.features(p, t, b, periods)[1].astype(np.float32)
    cached = vw.load(view[:4] + (1,))['x']
    v1 = transform(players, teams, ball, 'V1')
    v3 = transform(players, teams, ball, 'V3')
    out = {'H3_view': vw.name(view[:4] + (1,)), 'H3_equal': bool(np.array_equal(x, cached, equal_nan=True)),
           'V1_teams_unchanged': bool(np.array_equal(v1[1], teams)), 'V3_teams_swapped': bool(np.array_equal(v3[1], 1 - teams)),
           'V3_y_unchanged': bool(np.array_equal(v3[0][..., 1], players[..., 1], equal_nan=True))}
    out['ok'] = all(v for k, v in out.items() if k != 'H3_view')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'work/exp27/views-check.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    if not out['ok']:
        raise SystemExit('H3 failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        check()
        sys.exit(0)
    from concurrent.futures import ProcessPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    views = all_views()
    print(len(views), 'views x 4', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for n in pool.map(_cache, views):
            print('cached', n, flush=True)
    print('views ready')
