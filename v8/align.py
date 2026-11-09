"""Align a second tracking view (broadcast ATD) to the tactical timeline and fuse its ball observations.

Both views give player and ball positions in normalised pitch coordinates. Within a half, the two
recordings run at the same frame rate, so the offset delta = tactical_frame - broadcast_frame is
constant for live footage; replays and graphics do not match any tactical frame at that offset and
are rejected by residual checks. No identities, labels or clock values are used.

Algorithm
1. Orientation: the broadcast view may be mirrored relative to tactical; the best of 4 flips wins.
2. Global offset per tactical half: vote over sampled broadcast frames with a one-sided Chamfer
   distance (broadcast players -> nearest tactical players), then refine to a single frame.
3. Shots: contiguous broadcast segments with usable data; each shot is re-aligned locally around the
   half offset (replays may align to their true earlier time) and accepted only if its residual
   is small; per-frame residuals are checked again.
4. Fusion: tactical frames with no ball take the broadcast ball from an accepted frame.

Parallelism: orientation votes and shots are independent, so they run in a process pool (workers =
cpu count by default; `workers=1` runs in-process). Both views are converted to metres once and
shared with workers; results are identical whichever worker count is used.
"""
import multiprocessing as mp
import os

import numpy as np

PITCH = np.array([105.3, 68.])
FLIPS = ((False, False), (True, False), (False, True), (True, True))
MIN_PLAYERS = 5
CHAMFER_ACCEPT_M = 1.2        # mean nearest-player residual, metres
BALL_AGREE_M = 2.5
LOCAL_WINDOW = 2500            # frames searched around the half offset for each shot (100 s)
SHOT_MIN_FRAMES = 15
SHOT_MAX_FRAMES = 750
JUMP_M = 3.0                   # consecutive-frame Chamfer above this marks a cut


def _metres(xy, flip):
    out = xy.copy()
    if flip[0]:
        out[..., 0] = 1 - out[..., 0]
    if flip[1]:
        out[..., 1] = 1 - out[..., 1]
    return out * PITCH


def chamfer(pb, pt_batch):
    """pb: (k, 2) visible broadcast players; pt_batch: (n, m, 2) tactical players (NaN allowed).
    Returns (n,) mean over broadcast players of distance to the nearest tactical player."""
    d = np.linalg.norm(pb[None, :, None, :] - pt_batch[:, None, :, :], axis=3)  # (n, k, m)
    d = np.where(np.isfinite(d), d, np.inf)
    nearest = d.min(axis=2)
    nearest[~np.isfinite(nearest)] = 10.
    return np.minimum(nearest, 10.).mean(axis=1)


def visible_players(players, i):
    p = players[i]
    return p[np.isfinite(p).all(axis=1)]


def half_ranges(periods):
    out = {}
    for h in np.unique(periods[periods > 0]):
        idx = np.flatnonzero(periods == h)
        out[int(h)] = (int(idx[0]), int(idx[-1]) + 1)
    return out


def global_offset(tact_m, bcast_m, half, sample=100, stride=10):
    """Vote for delta over sampled broadcast frames against one tactical half. Both views in metres
    (broadcast already flipped). Returns (delta, score)."""
    a, b = half
    t_idx = np.arange(a, b, stride)
    pt = tact_m[t_idx]
    votes = {}
    for i in range(0, len(bcast_m), sample):
        pb = visible_players(bcast_m, i)
        if len(pb) < MIN_PLAYERS:
            continue
        c = chamfer(pb, pt)
        j = int(np.argmin(c))
        if c[j] < CHAMFER_ACCEPT_M:
            delta = int(t_idx[j]) - i
            key = delta // (2 * stride)
            votes[key] = votes.get(key, 0.) + np.exp(-c[j])
    if not votes:
        return None, 0.
    key = max(votes, key=votes.get)
    coarse = key * 2 * stride + stride
    # refine: evaluate deltas in a +-2*stride window on all sampled frames that fall in the half
    best, best_cost = coarse, np.inf
    for delta in range(coarse - 2 * stride, coarse + 2 * stride + 1):
        costs = []
        for i in range(0, len(bcast_m), sample):
            t = i + delta
            if not a <= t < b:
                continue
            pb = visible_players(bcast_m, i)
            if len(pb) >= MIN_PLAYERS:
                costs.append(chamfer(pb, tact_m[t:t + 1])[0])
        if len(costs) >= 5:
            cost = float(np.median(costs))
            if cost < best_cost:
                best, best_cost = delta, cost
    return best, votes[key]


def shots(bcast, min_frames=SHOT_MIN_FRAMES, jump_m=JUMP_M, max_frames=SHOT_MAX_FRAMES):
    """Contiguous usable segments, split at cuts (configuration jumps) and capped in length."""
    ok = np.isfinite(bcast).all(axis=2).sum(axis=1) >= MIN_PLAYERS
    m = bcast * PITCH
    jump = np.zeros(len(bcast), bool)
    for i in range(1, len(bcast)):
        if ok[i] and ok[i - 1]:
            pb = m[i][np.isfinite(m[i]).all(axis=1)]
            prev = m[i - 1][np.isfinite(m[i - 1]).all(axis=1)]
            d = np.linalg.norm(pb[:, None, :] - prev[None, :, :], axis=2).min(axis=1)
            jump[i] = np.median(d) > jump_m  # a cut moves every player; flicker moves a few
    breaks = np.r_[0, np.flatnonzero((ok[1:] != ok[:-1]) | jump[1:]) + 1, len(ok)]
    out = []
    for s, e in zip(breaks[:-1], breaks[1:]):
        if not ok[s]:
            continue
        for a in range(s, e, max_frames):
            b = min(e, a + max_frames)
            if b - a >= min_frames:
                out.append((int(a), int(b)))
    return out


def align_shot(tact_m, bcast_m, shot, delta0, half, sample=5):
    """Local search of delta around delta0 for one shot (both views in metres).
    Returns (delta, mean residual) or (None, inf)."""
    a, b = half
    s, e = shot
    frames = list(range(s, e, sample))
    pbs = [visible_players(bcast_m, i) for i in frames]
    keep = [k for k, p in enumerate(pbs) if len(p) >= MIN_PLAYERS]
    if len(keep) < 3:
        return None, np.inf
    frames, pbs = [frames[k] for k in keep], [pbs[k] for k in keep]
    lo, hi = delta0 - LOCAL_WINDOW, delta0 + LOCAL_WINDOW
    deltas = np.arange(lo, hi + 1)
    total = np.zeros(len(deltas))
    count = np.zeros(len(deltas))
    for i, pb in zip(frames, pbs):
        t = i + deltas
        valid = (t >= a) & (t < b)
        if not valid.any():
            continue
        c = chamfer(pb, tact_m[t[valid]])
        total[valid] += c
        count[valid] += 1
    enough = count >= max(3, .5 * len(frames))
    if not enough.any():
        return None, np.inf
    mean = np.where(enough, total / np.maximum(count, 1), np.inf)
    j = int(np.argmin(mean))
    return int(deltas[j]), float(mean[j])


_SHARED = {}


def _init(tact_m, bcast_views):
    _SHARED['tact_m'], _SHARED['views'] = tact_m, bcast_views


def _vote(job):
    flip, h, rng, sample, stride = job
    return flip, h, global_offset(_SHARED['tact_m'], _SHARED['views'][flip], rng, sample, stride)


def _shot(job):
    """Align one shot to the best half; return accepted (frame, tactical index, residual) triples."""
    flip, shot, offsets, halves = job
    tact_m, bcast_m = _SHARED['tact_m'], _SHARED['views'][flip]
    cands = []
    for h, rng in halves.items():
        if offsets[h] is None:
            continue
        d, r = align_shot(tact_m, bcast_m, shot, offsets[h], rng)
        if d is not None:
            cands.append((r, d, rng))
    if not cands:
        return []
    r, d, (a, b) = min(cands)
    if r > CHAMFER_ACCEPT_M:
        return []
    out = []
    for i in range(*shot):
        t = i + d
        if not a <= t < b:
            continue
        pb = bcast_m[i][np.isfinite(bcast_m[i]).all(axis=1)]
        if len(pb) < MIN_PLAYERS:
            continue
        c = chamfer(pb, tact_m[t:t + 1])[0]
        if c <= CHAMFER_ACCEPT_M:
            out.append((i, t, c))
    return out


def _run(pool, fn, jobs):
    return list(map(fn, jobs)) if pool is None else pool.map(fn, jobs, chunksize=1)


def align(tact, bcast, periods, sample=100, stride=10, workers=None):
    """Returns dict with per-broadcast-frame tactical index (-1 if unaligned), residuals, flip, offsets."""
    halves = half_ranges(periods)
    tact_m = _metres(tact, (False, False))
    views = {flip: _metres(bcast, flip) for flip in FLIPS}
    workers = os.cpu_count() or 1 if workers is None else workers
    pool = None
    if workers > 1:
        ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
        pool = ctx.Pool(workers, initializer=_init, initargs=(tact_m, views))
    else:
        _init(tact_m, views)
    try:
        votes = _run(pool, _vote, [(flip, h, rng, sample, stride) for flip in FLIPS for h, rng in halves.items()])
        scores, offsets = {flip: 0. for flip in FLIPS}, {flip: {} for flip in FLIPS}
        for flip, h, (d, s) in votes:
            offsets[flip][h] = d
            scores[flip] += s
        flip = max(FLIPS, key=lambda f: scores[f])   # first flip wins ties, as before
        offsets = offsets[flip]
        mapping = np.full(len(bcast), -1)
        residual = np.full(len(bcast), np.inf)
        for triples in _run(pool, _shot, [(flip, shot, offsets, halves) for shot in shots(bcast)]):
            for i, t, c in triples:
                mapping[i], residual[i] = t, c
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {'mapping': mapping, 'residual': residual, 'flip': flip, 'offsets': offsets}


def fuse_ball(tact_ball, bcast_ball, alignment):
    """Fill missing tactical ball frames from aligned broadcast frames (lowest residual wins).
    Returns (fused_ball, source) with source 0 = tactical, 1 = broadcast, -1 = still missing."""
    flip = alignment['flip']
    bb = bcast_ball.copy()
    if flip[0]:
        bb[:, 0] = 1 - bb[:, 0]
    if flip[1]:
        bb[:, 1] = 1 - bb[:, 1]
    fused = tact_ball.copy()
    source = np.where(np.isfinite(tact_ball).all(axis=1), 0, -1)
    best = np.full(len(tact_ball), np.inf)
    m, r = alignment['mapping'], alignment['residual']
    for i in np.flatnonzero((m >= 0) & np.isfinite(bb).all(axis=1)):
        t = m[i]
        if source[t] == 0 or r[i] >= best[t]:
            continue
        fused[t], source[t], best[t] = bb[i], 1, r[i]
    return fused, source


def agreement(tact_ball, bcast_ball, alignment):
    """Median distance (m) between the two balls where both are visible on aligned frames; count."""
    flip = alignment['flip']
    bb = bcast_ball.copy()
    if flip[0]:
        bb[:, 0] = 1 - bb[:, 0]
    if flip[1]:
        bb[:, 1] = 1 - bb[:, 1]
    m = alignment['mapping']
    ok = (m >= 0) & np.isfinite(bb).all(axis=1)
    ok[ok] &= np.isfinite(tact_ball[m[ok]]).all(axis=1)
    if not ok.any():
        return None, 0
    d = np.linalg.norm((bb[ok] - tact_ball[m[ok]]) * PITCH, axis=1)
    return float(np.median(d)), int(ok.sum())
