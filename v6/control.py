"""Ball-control features (EXPERIMENT-10). Thresholds fixed a priori; no labels, identities or clock."""
import numpy as np

PITCH = np.array([105., 68.])
FPS = 25
CONTROL_DIST, CONTROL_REL_SPEED = 1.5, 2.5
WINDOWS = (12, 50, 150)
CAP = 60.


def _halves(periods):
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if periods[a] > 0]


def velocity(xy, periods):
    """Centred 5-frame difference in m/s; NaN where a neighbour is missing or crosses a half."""
    v = np.full(xy.shape, np.nan)
    for a, b in _halves(periods):
        seg = xy[a:b] * PITCH
        if b - a > 4:
            v[a + 2:b - 2] = (seg[4:] - seg[:-4]) * FPS / 4.
    return v


def roll_mean(values, window, periods):
    out = np.zeros(len(values))
    for a, b in _halves(periods):
        s = np.r_[0., np.cumsum(values[a:b])]
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - window), np.minimum(b - a, i + window + 1)
        out[a:b] = (s[hi] - s[lo]) / (hi - lo)
    return out


def since(flag, periods, backward=False):
    out = np.full(len(flag), CAP)
    for a, b in _halves(periods):
        f = flag[a:b][::-1] if backward else flag[a:b]
        i = np.arange(len(f))
        last = np.maximum.accumulate(np.where(f, i, -1))
        t = np.where(last >= 0, (i - last) / FPS, CAP)
        out[a:b] = np.minimum(t[::-1] if backward else t, CAP)
    return out


def carry_team(team, periods, backward=False):
    out = np.zeros(len(team))
    for a, b in _halves(periods):
        t = team[a:b][::-1] if backward else team[a:b]
        i = np.arange(len(t))
        idx = np.maximum.accumulate(np.where(t != 0, i, -1))
        c = np.where(idx >= 0, t[np.maximum(idx, 0)], 0.)
        out[a:b] = c[::-1] if backward else c
    return out


def control_features(players, teams, ball, periods):
    n = len(periods)
    visible = np.isfinite(ball).all(axis=1)
    d = np.linalg.norm((players - ball[:, None, :]) * PITCH, axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    k = np.argmin(d, axis=1)
    d1 = d[np.arange(n), k]
    vb = velocity(ball, periods)
    vp = velocity(players, periods)
    rel = np.linalg.norm(vb - vp[np.arange(n), k], axis=1)
    controlled = visible & (d1 <= CONTROL_DIST) & np.isfinite(rel) & (rel <= CONTROL_REL_SPEED) & (periods > 0)
    team = np.where(controlled, np.where(teams[k] == 0, 1., -1.), 0.)
    home, away = (team > 0).astype(float), (team < 0).astype(float)
    cols = []
    for w in WINDOWS:
        cols += [roll_mean(home, w, periods), roll_mean(away, w, periods)]
    cols += [since(home > 0, periods), since(away > 0, periods),
             since(home > 0, periods, backward=True), since(away > 0, periods, backward=True),
             carry_team(team, periods), carry_team(team, periods, backward=True)]
    return np.column_stack(cols)
