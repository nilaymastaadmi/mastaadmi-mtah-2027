"""EXPERIMENT-22: 5 fps dilated conv net with +-25.4 s context (work/EXPERIMENT-22.md, binding spec).

    OMP_NUM_THREADS=3 .venv-video/bin/python -m exp22.net --arm M            # folds 0..4, resumable per fold
    OMP_NUM_THREADS=3 .venv-video/bin/python -m exp22.net --arm M --final    # all training matches, + PFF q
    arms: M (with Metrica), A2 (no Metrica)

numpy + torch only. Reads view caches via exp22.views. Writes q and c (25 fps, float32) per held-out view to
external/exp22/<arm>/<view>.npz, models to external/exp22/<arm>/models/, log to work/exp22/train-<arm>.log.
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
from exp18.arm_a import transform, halves  # noqa: E402  (same scaling and half split as arm A)
from exp18.targets import change_events, near  # noqa: E402
from exp22 import views as vw  # noqa: E402

OUT = ROOT / 'external/exp22'
SEED = 22
BLOCK, OFFSET = 5, 2
CHUNK, BATCH, EPOCHS = 512, 16, 12
LR, WD = 2e-3, 1e-4
WIDTH, DILATIONS, DROPOUT = 64, (1, 2, 4, 8, 16, 32, 64), .1
BOUNDARY_K, BOUNDARY_W = 25, 2.                      # +-1 s at 25 fps
SHORT = ((75, 2.), (150, 1.5))                       # raw H/A run < 3 s: 2, < 6 s: 1.5
VIEW_W = {'sc': .25}
CHANGE_POS_W, CHANGE_LOSS_W = 5., .5
THREADS = 3
CLASSES = {'H': 0, 'A': 1, 'D': 2}


# ---------- scaling ----------

def fit_scaler(views):
    rows = []
    for v in views:
        d = vw.load(v)
        rows.append(d['x'][d['periods'] > 0][::10])
    x = np.concatenate(rows).astype(np.float64)
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-9), iqr, 1.)
    med = np.where(np.isfinite(med), med, 0.)
    nan_cols = np.flatnonzero(np.isnan(x).any(axis=0))
    return {'med': med.astype(np.float32), 'iqr': iqr.astype(np.float32), 'nan_cols': nan_cols.astype(np.int64)}


# ---------- 5 fps blocks ----------

def block_index(a, b):
    """Block starts and centres for frames [a, b): non-overlapping blocks of 5 from the half's first frame."""
    starts = np.arange(a, b, BLOCK)
    ends = np.minimum(starts + BLOCK, b)
    centres = np.minimum(starts + OFFSET, ends - 1)
    return starts, ends, centres


def block_mean(z, a, b):
    starts, ends, _ = block_index(a, b)
    s = np.add.reduceat(z[a:b], starts - a, axis=0)
    return (s / (ends - starts)[:, None]).astype(np.float32)


def short_weight(labels, periods):
    """Weight per frame from the length of the raw H/A run it sits in (runs broken by D, X and halves)."""
    w = np.ones(len(labels), np.float32)
    lab = np.where(np.asarray(periods) > 0, labels, 'X')
    edges = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1, len(lab)]
    for a, b in zip(edges[:-1], edges[1:]):
        if lab[a] in ('H', 'A'):
            for limit, weight in SHORT:
                if b - a < limit:
                    w[a:b] = weight
                    break
    return w


def frame_targets(d, ds):
    labels, periods = d['labels'], d['periods']
    y = np.array([CLASSES.get(l, -1) for l in labels], np.int64)
    y[periods <= 0] = -1
    cm = y >= 0                                           # change loss: every non-X frame
    if ds == 'sc':
        y = y.copy()
        y[~d['visible'].astype(bool)] = -1                # class loss: visible-ball frames only (as v8)
    ev = change_events(labels)
    w = np.where(near(ev, BOUNDARY_K, periods), BOUNDARY_W, 1.).astype(np.float32)
    w *= short_weight(labels, periods) * VIEW_W.get(ds, 1.)
    return y, w, ev, cm


def build_halves(views, scaler):
    out = []
    for v in views:
        d = vw.load(v)
        z = transform(d['x'], scaler)
        y, w, ev, cm = frame_targets(d, v[0])
        vwt = VIEW_W.get(v[0], 1.)
        for a, b in halves(d['periods']):
            starts, ends, centres = block_index(a, b)
            if len(starts) < CHUNK:
                continue
            xb = block_mean(z, a, b)
            has = np.add.reduceat(ev[a:b].astype(np.int64), starts - a) > 0
            chg = has.copy()
            chg[1:] |= has[:-1]
            chg[:-1] |= has[1:]
            out.append((torch.from_numpy(np.ascontiguousarray(xb.T)), torch.from_numpy(y[centres]),
                        torch.from_numpy(w[centres]), torch.from_numpy(chg.astype(np.float32)),
                        torch.from_numpy((cm[centres] * vwt).astype(np.float32))))
    return out


# ---------- model ----------

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.conv = nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d)
        self.pw = nn.Conv1d(WIDTH, WIDTH, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
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


def receptive_field():
    return 1 + 2 * sum(DILATIONS)                         # 255 blocks = +-127 blocks = +-25.4 s


def setup():
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def loss_of(model, batch):
    x, y, w, c, cm = batch
    logits, chg = model(x)
    ce = nn.functional.cross_entropy(logits, y.clamp(min=0), reduction='none')
    m = (y >= 0).float() * w
    ce = (ce * m).sum() / m.sum().clamp(min=1e-6)
    bce = nn.functional.binary_cross_entropy_with_logits(chg, c, pos_weight=torch.tensor(CHANGE_POS_W),
                                                         reduction='none')
    bce = (bce * cm).sum() / cm.sum().clamp(min=1e-6)
    return ce + CHANGE_LOSS_W * bce, ce.item(), bce.item()


def train(views, seed_index, log, tag):
    setup()
    scaler = fit_scaler(views)
    data = build_halves(views, scaler)
    lengths = np.array([len(h[1]) for h in data])
    total = int(lengths.sum())
    steps = (total // CHUNK) // BATCH
    c_in = data[0][0].shape[0]
    model = Net(c_in)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps)
    log(f'{tag}: {len(views)} views, {len(data)} halves, {total} blocks, {c_in} channels, {steps} steps/epoch, '
        f'{sum(p.numel() for p in model.parameters())} parameters, receptive field {receptive_field()} blocks')
    prob = lengths / lengths.sum()
    for epoch in range(EPOCHS):
        rng = np.random.default_rng([SEED, seed_index, epoch])
        model.train()
        t0, tot, tce, tbce = time.time(), 0., 0., 0.
        for _ in range(steps):
            idx = rng.choice(len(data), size=BATCH, p=prob)
            starts = [int(rng.integers(0, lengths[i] - CHUNK + 1)) for i in idx]
            batch = [torch.stack([data[i][j][..., s:s + CHUNK] for i, s in zip(idx, starts)]) for j in range(5)]
            loss, ce, bce = loss_of(model, batch)
            if not np.isfinite(loss.item()):
                raise SystemExit(f'{tag}: non-finite loss at epoch {epoch}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, tce, tbce = tot + loss.item(), tce + ce, tbce + bce
        n = max(1, steps)
        log(f'{tag} epoch {epoch}: loss {tot / n:.4f} ce {tce / n:.4f} bce {tbce / n:.4f} ({time.time() - t0:.0f} s)')
    return model, scaler


@torch.no_grad()
def predict(model, scaler, x, periods):
    """q (N x 3) and c (N) at 25 fps: block outputs interpolated linearly between block centres."""
    model.eval()
    z = transform(x, scaler)
    q = np.full((len(periods), 3), 1 / 3, np.float32)
    c = np.zeros(len(periods), np.float32)
    for a, b in halves(periods):
        _, _, centres = block_index(a, b)
        logits, chg = model(torch.from_numpy(np.ascontiguousarray(block_mean(z, a, b).T))[None])
        qb = torch.softmax(logits[0].double(), dim=0).T.numpy()
        cb = torch.sigmoid(chg[0].double()).numpy()
        frames = np.arange(a, b)
        qi = np.column_stack([np.interp(frames, centres, qb[:, k]) for k in range(3)])
        q[a:b] = qi / qi.sum(axis=1, keepdims=True)
        c[a:b] = np.interp(frames, centres, cb)
    return q, c


def save_model(path, model, scaler):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': model.state_dict(), 'c_in': model.inp.in_channels,
                **{k: torch.from_numpy(np.asarray(v)) for k, v in scaler.items()}}, path)


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


def write_preds(arm_dir, model, scaler, views, out_name):
    digests = {}
    for v in views:
        d = vw.load(v)
        q, c = predict(model, scaler, d['x'], d['periods'])
        if not (np.isfinite(q).all() and np.allclose(q.sum(1), 1, atol=1e-5) and (q >= 0).all()):
            raise SystemExit(f'{out_name(v)}: q out of bounds')
        tmp = arm_dir / f'{out_name(v)}.tmp.npz'
        np.savez(tmp, q=q, c=c)
        tmp.rename(arm_dir / f'{out_name(v)}.npz')
        digests[out_name(v)] = digest(q, c)
    return digests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=('M', 'A2'), required=True)
    ap.add_argument('--fold', type=int)
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    arm_dir = OUT / args.arm
    (arm_dir / 'models').mkdir(parents=True, exist_ok=True)
    logf = ROOT / f'work/exp22/train-{args.arm}.log'
    metrica = args.arm == 'M'

    def log(msg):
        line = f'{time.strftime("%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with logf.open('a') as f:
            f.write(line + '\n')

    if args.final:
        done = arm_dir / 'final.done'
        if done.exists():
            return
        path = arm_dir / 'models/final.pt'
        if path.exists():
            model, scaler = load_model(path)
            log('final: resuming from saved model (predictions only)')
        else:
            t0 = time.time()
            model, scaler = train(vw.train_views(None, metrica), 5, log, f'{args.arm} final')
            save_model(path, model, scaler)
            log(f'{args.arm} final trained in {time.time() - t0:.0f} s')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (arm_dir / 'models/final.sha256').write_text(sha + '\n')
        digests = write_preds(arm_dir, model, scaler, vw.pff_views(), lambda v: f'pff-{v[1]}')
        (arm_dir / 'final.digests.json').write_text(json.dumps(digests, indent=1) + '\n')
        done.touch()
        log(f'{args.arm} final: model sha256 {sha[:16]}, {len(digests)} PFF views predicted (labels not read)')
        return
    for fold in ([args.fold] if args.fold is not None else range(5)):
        done = arm_dir / f'fold{fold}.done'
        if done.exists():
            continue
        t0 = time.time()
        model, scaler = train(vw.train_views(fold, metrica), fold, log, f'{args.arm} fold {fold}')
        save_model(arm_dir / f'models/fold{fold}.pt', model, scaler)
        digests = write_preds(arm_dir, model, scaler, vw.eval_views(fold), vw.name)
        (arm_dir / f'fold{fold}.digests.json').write_text(json.dumps(digests, indent=1) + '\n')
        done.touch()
        log(f'{args.arm} fold {fold} done in {time.time() - t0:.0f} s, {len(digests)} views')


if __name__ == '__main__':
    main()
