"""EXPERIMENT-18 arm A: dilated temporal conv net over per-frame inputs (work/EXPERIMENT-18.md, binding spec).

    OMP_NUM_THREADS=6 .venv-video/bin/python -m exp18.arm_a              # folds 0..4, resumable
    OMP_NUM_THREADS=6 .venv-video/bin/python -m exp18.arm_a --fold 0
    OMP_NUM_THREADS=6 .venv-video/bin/python -m exp18.arm_a --final      # all 27 matches, for v10

Reads only the float32 caches written by `exp18.views` (external/exp18/views/), so it needs numpy and torch only.
Writes q (H/A/D softmax) and c (change sigmoid) per held-out view to external/exp18/arm_a/, models to
external/exp18/arm_a/models/, and the training log to work/exp18/arm_a_train.log.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp18.targets import change_events, near  # noqa: E402

CACHE = ROOT / 'external/exp18/views'
OUT = ROOT / 'external/exp18/arm_a'
LOG = ROOT / 'work/exp18/arm_a_train.log'
FOLDS = 5
SEED = 18
CHUNK, BATCH, EPOCHS = 2048, 16, 12
LR, WD = 2e-3, 1e-4
WIDTH, DILATIONS, DROPOUT = 64, (1, 2, 4, 8, 16, 32, 48), .1
BOUNDARY_K, BOUNDARY_W = 25, 2.
CHANGE_K, CHANGE_POS_W, CHANGE_LOSS_W = 6, 5., .5
CLIP = 10.
THREADS = 6
CLASSES = {'H': 0, 'A': 1, 'D': 2}


# ---------- views (same lists as exp18.views, without its sklearn-importing dependencies) ----------

def _ids():
    names = [p.name[:-4] for p in CACHE.glob('*.npz') if not p.name.endswith('.tmp.npz')]
    dfl = sorted({n.split('-')[1] for n in names if n.startswith('dfl-')})
    sc = sorted({int(n.split('-')[1]) for n in names if n.startswith('sc-')})
    if (len(dfl), len(sc)) != (7, 20):   # a partial cache would shift fold_of (review fix, 7 Nov)
        raise SystemExit(f'view cache incomplete: {len(dfl)} DFL, {len(sc)} SkillCorner; run exp18.views')
    return dfl, sc


def fold_of(ids, m):
    return sorted(ids).index(m) % FOLDS          # identical to v4.develop.fold_of


def train_views(fold):
    """fold None = all matches (final model)."""
    dfl, sc = _ids()
    out = [('dfl', m, c, 'train') for m in dfl if fold is None or fold_of(dfl, m) != fold
           for c in ('natural', 'gaps', 'contested')]
    return out + [('sc', m, 'real', 'eval') for m in sc if fold is None or fold_of(sc, m) != fold]


def eval_views(fold):
    dfl, sc = _ids()
    out = [('sc', m, 'real', 'eval') for m in sc if fold_of(sc, m) == fold]
    return out + [('dfl', m, c, 'eval') for c in ('natural', 'contested') for m in dfl if fold_of(dfl, m) == fold]


def name(view):
    return '-'.join(str(p) for p in view)


def load(view):
    with np.load(CACHE / f'{name(view)}.npz') as z:
        return {k: z[k] for k in z.files}


def halves(periods):
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if periods[a] > 0]


# ---------- input scaling ----------

def fit_scaler(views):
    rows = []
    for v in views:
        d = load(v)
        rows.append(d['x'][d['periods'] > 0][::10])
    x = np.concatenate(rows).astype(np.float64)
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-9), iqr, 1.)
    med = np.where(np.isfinite(med), med, 0.)
    nan_cols = np.flatnonzero(np.isnan(x).any(axis=0))
    return {'med': med.astype(np.float32), 'iqr': iqr.astype(np.float32), 'nan_cols': nan_cols.astype(np.int64)}


def transform(x, scaler):
    z = np.clip((x - scaler['med']) / scaler['iqr'], -CLIP, CLIP)
    miss = np.isnan(x[:, scaler['nan_cols']]).astype(np.float32)
    z = np.where(np.isnan(z), 0., z).astype(np.float32)
    return np.hstack([z, miss])


def targets(d, ds):
    labels, periods = d['labels'], d['periods']
    y = np.array([CLASSES.get(l, -1) for l in labels], np.int64)
    y[periods <= 0] = -1
    ychg = y.copy()                                           # change loss: every non-X frame
    if ds == 'sc':
        y[~d['visible']] = -1                                 # class loss: visible-ball frames only (as v8)
    ev = change_events(labels)
    w = np.where(near(ev, BOUNDARY_K, periods), BOUNDARY_W, 1.).astype(np.float32)
    c = near(ev, CHANGE_K, periods).astype(np.float32)
    return y, w, c, ychg >= 0


# ---------- model ----------

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.conv = nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d)
        self.pw = nn.Conv1d(WIDTH, WIDTH, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):                                     # x: B x C x T
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.pw(self.act(self.conv(h))))


class Net(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.inp = nn.Conv1d(c_in, WIDTH, 1)
        self.blocks = nn.Sequential(*[Block(d) for d in DILATIONS])
        self.norm = nn.LayerNorm(WIDTH)
        self.cls = nn.Conv1d(WIDTH, 3, 1)
        self.chg = nn.Conv1d(WIDTH, 1, 1)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return self.cls(h), self.chg(h)[:, 0]


def setup():
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


# ---------- training ----------

def build_halves(views, scaler):
    out = []
    for v in views:
        d = load(v)
        z = transform(d['x'], scaler)
        y, w, c, cm = targets(d, v[0])
        for a, b in halves(d['periods']):
            if b - a >= CHUNK:
                out.append((torch.from_numpy(np.ascontiguousarray(z[a:b].T)), torch.from_numpy(y[a:b]),
                            torch.from_numpy(w[a:b]), torch.from_numpy(c[a:b]), torch.from_numpy(cm[a:b])))
    return out


def loss_of(model, batch):
    x, y, w, c, cm = batch
    logits, chg = model(x)
    ce = nn.functional.cross_entropy(logits, y.clamp(min=0), reduction='none')
    m = (y >= 0).float() * w
    ce = (ce * m).sum() / m.sum().clamp(min=1.)
    bce = nn.functional.binary_cross_entropy_with_logits(chg, c, pos_weight=torch.tensor(CHANGE_POS_W),
                                                         reduction='none')
    cmf = cm.float()
    bce = (bce * cmf).sum() / cmf.sum().clamp(min=1.)
    return ce + CHANGE_LOSS_W * bce, ce.item(), bce.item()


def train(fold, log):
    setup()
    views = train_views(fold)
    scaler = fit_scaler(views)
    data = build_halves(views, scaler)
    lengths = np.array([len(h[1]) for h in data])
    total = int(lengths.sum())
    chunks_per_epoch = total // CHUNK
    steps_per_epoch = chunks_per_epoch // BATCH
    c_in = data[0][0].shape[0]
    model = Net(c_in)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    log(f'fold {fold}: {len(views)} views, {len(data)} halves, {total} frames, {c_in} input channels, '
        f'{steps_per_epoch} steps/epoch, {sum(p.numel() for p in model.parameters())} parameters')
    prob = lengths / lengths.sum()
    for epoch in range(EPOCHS):
        rng = np.random.default_rng([SEED, FOLDS if fold is None else fold, epoch])   # final model: 5 (crash fix, 7 Nov)
        model.train()
        t0, tot, tce, tbce = time.time(), 0., 0., 0.
        for step in range(steps_per_epoch):
            idx = rng.choice(len(data), size=BATCH, p=prob)
            starts = [int(rng.integers(0, lengths[i] - CHUNK + 1)) for i in idx]
            batch = [torch.stack([data[i][j][..., s:s + CHUNK] for i, s in zip(idx, starts)]) for j in range(5)]
            loss, ce, bce = loss_of(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, tce, tbce = tot + loss.item(), tce + ce, tbce + bce
        n = max(1, steps_per_epoch)
        log(f'fold {fold} epoch {epoch}: loss {tot / n:.4f} ce {tce / n:.4f} bce {tbce / n:.4f} '
            f'({time.time() - t0:.0f} s)')
    return model, scaler


# ---------- inference ----------

@torch.no_grad()
def predict(model, scaler, x, periods):
    """q (N x 3 softmax H/A/D) and c (N change probability); uniform q and c = 0 outside play."""
    model.eval()
    z = transform(x, scaler)
    q = np.full((len(periods), 3), 1 / 3, np.float32)
    c = np.zeros(len(periods), np.float32)
    for a, b in halves(periods):
        logits, chg = model(torch.from_numpy(np.ascontiguousarray(z[a:b].T))[None])
        q[a:b] = torch.softmax(logits[0], dim=0).T.numpy()
        c[a:b] = torch.sigmoid(chg[0]).numpy()
    return q, c


def save_model(path, model, scaler, c_in):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': model.state_dict(), 'c_in': c_in, **{k: torch.from_numpy(np.asarray(v))
                                                               for k, v in scaler.items()}}, path)


def load_model(path):
    blob = torch.load(path, weights_only=True)
    setup()
    model = Net(int(blob['c_in']))
    model.load_state_dict(blob['state'])
    return model, {k: blob[k].numpy() for k in ('med', 'iqr', 'nan_cols')}


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with LOG.open('a') as f:
            f.write(line + '\n')

    if args.final:
        path = OUT / 'models/final.pt'
        if not path.exists():
            model, scaler = train(None, log)
            save_model(path, model, scaler, model.inp.in_channels)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (OUT / 'models/final.sha256').write_text(sha + '\n')
        log(f'final model sha256 {sha}')
        return
    folds = [args.fold] if args.fold is not None else range(FOLDS)
    for fold in folds:
        done = OUT / f'fold{fold}.done'
        if done.exists():
            continue
        t0 = time.time()
        model, scaler = train(fold, log)
        save_model(OUT / f'models/fold{fold}.pt', model, scaler, model.inp.in_channels)
        digests = {}
        for v in eval_views(fold):
            d = load(v)
            q, c = predict(model, scaler, d['x'], d['periods'])
            tmp = OUT / f'{name(v)}.tmp.npz'
            np.savez(tmp, q=q, c=c)
            tmp.rename(OUT / f'{name(v)}.npz')
            digests[name(v)] = digest(q, c)
        (OUT / f'fold{fold}.digests.json').write_text(json.dumps(digests, indent=1) + '\n')
        done.touch()
        log(f'fold {fold} done in {time.time() - t0:.0f} s, {len(digests)} views')


if __name__ == '__main__':
    main()
