"""Simulated broadcast-like second view for EXPERIMENT-13 analogue evaluation and tests."""
import numpy as np

from v8 import align as al


def second_view(players, ball, periods, delta=1234, mirror=(True, False), seed=1, radius=35.,
                noise=.15, ball_visible=.7, live_share=.7):
    """Frame i shows tactical frame i+delta in live shots; replays show earlier footage frame by
    frame (jumpy, so they fragment); only players within `radius` m of the ball are visible, with
    10% dropout; ball visible in `ball_visible` of live frames; Gaussian noise; optional mirror."""
    rng = np.random.default_rng(seed)
    n = len(players) - delta - 300
    bp = np.full((n, players.shape[1], 2), np.nan)
    bb = np.full((n, 2), np.nan)
    truth = np.full(n, -1)
    i = 0
    while i < n:
        live = rng.random() < live_share
        length = int(rng.integers(250, 1000)) if live else int(rng.integers(100, 400))
        back = int(rng.integers(600, 2000))
        for k in range(i, min(n, i + length)):
            t = k + delta if live else max(0, k + delta - back)
            if t >= len(players):
                break
            if live and periods[t] > 0:
                truth[k] = t
            p = players[t]
            if not np.isfinite(ball[t]).all():
                continue
            near = np.linalg.norm((p - ball[t]) * al.PITCH, axis=1) < radius
            vis = near & (rng.random(len(p)) > .1)
            bp[k, vis] = p[vis] + rng.normal(0, noise, (vis.sum(), 2)) / al.PITCH
            if rng.random() < ball_visible:
                bb[k] = ball[t] + rng.normal(0, noise, 2) / al.PITCH
        i += length
    if mirror[0]:
        bp[..., 0], bb[:, 0] = 1 - bp[..., 0], 1 - bb[:, 0]
    if mirror[1]:
        bp[..., 1], bb[:, 1] = 1 - bp[..., 1], 1 - bb[:, 1]
    return bp, bb, truth
