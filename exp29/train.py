"""EXPERIMENT-29 trainings: recipe M (exp22.net.train, unchanged) with PFF-64 as a fourth provider (work/EXPERIMENT-29.md).

    OMP_NUM_THREADS=3 .venv-video/bin/python -B -m exp29.train H1 sc-with-A ...    # runs in order, resumable per run

Runs: H1 (no PFF, nothing held out: must reproduce v12/final.pt), sc-without, dfl-without, sc-with-A, sc-with-B,
dfl-with-A, dfl-with-B, final-A, final-B. A: PFF weight 0.25, B: 1.0. Every run uses seed index 5. PFF views: the
cache (external/pff-cache/views) unswapped; swapped = EXPERIMENT-27's V2 view (external/exp27/views/pff-<m>-V2.npz)
with H and A exchanged. Models to external/exp29/models/<run>.pt, logs to work/exp29/train-<run>.log.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp22 import net, views as vw  # noqa: E402

MODELS = ROOT / 'external/exp29/models'
LOGS = ROOT / 'work/exp29'
SWAPPED = ROOT / 'external/exp27/views'
SEED_INDEX = 5
WEIGHTS = {'A': .25, 'B': 1.}
RUNS = {'H1': (None, None), 'sc-without': ('sc', None), 'dfl-without': ('dfl', None)}
for _c, _w in WEIGHTS.items():
    RUNS.update({f'sc-with-{_c}': ('sc', _w), f'dfl-with-{_c}': ('dfl', _w), f'final-{_c}': (None, _w)})
_load = vw.load


def swap_labels(labels):
    out = np.array(labels, dtype='<U1').copy()
    h, a = out == 'H', out == 'A'
    out[h], out[a] = 'A', 'H'
    return out


def load(view):
    """exp22.views.load plus swapped PFF views (exp22.views.load ignores the swap flag for PFF)."""
    if view[0] != 'pff' or not view[4]:
        return _load(view)
    d = _load(view[:4] + (0,))
    with np.load(SWAPPED / f'pff-{view[1]}-V2.npz') as z:
        x, periods = z['x'], z['periods']
    if not np.array_equal(periods, d['periods']):
        raise SystemExit(f'pff-{view[1]}: V2 periods differ from the cache')
    return {'x': x, 'labels': swap_labels(d['labels']), 'periods': periods}


def train_views(held_out, w):
    views = [v for v in vw.train_views(None, True) if v[0] != held_out]
    if w is not None:
        views += [('pff', m, 'real', 'eval', s) for m in sorted(v[1] for v in vw.pff_views()) for s in (0, 1)]
    return views


def run(name):
    held, w = RUNS[name]
    path = MODELS / f'{name}.pt'
    if path.exists():
        return
    vw.load = load                                       # exp22.net reads views through exp22.views.load
    if w is not None:
        net.VIEW_W['pff'] = w                            # as VIEW_W['sc'] = 0.25: class-loss weight and change mask
    logf = LOGS / f'train-{name}.log'

    def log(msg):
        line = f'{time.strftime("%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with logf.open('a') as f:
            f.write(line + '\n')

    t0 = time.time()
    views = train_views(held, w)
    log(f'{name}: held out {held}, PFF weight {w}, {len(views)} views ({sum(v[0] == "pff" for v in views)} PFF)')
    model, scaler = net.train(views, SEED_INDEX, log, name)
    tmp = path.with_suffix('.tmp.pt')
    net.save_model(tmp, model, scaler)
    tmp.rename(path)
    log(f'{name} trained in {time.time() - t0:.0f} s, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}')
    net.VIEW_W.pop('pff', None)


if __name__ == '__main__':
    LOGS.mkdir(parents=True, exist_ok=True)
    for r in sys.argv[1:]:
        if r not in RUNS:
            raise SystemExit(f'unknown run {r}')
    for r in sys.argv[1:]:
        run(r)
