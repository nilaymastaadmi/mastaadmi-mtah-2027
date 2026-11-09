"""Short-gap ball interpolation (EXPERIMENT-15): fill missing ball frames of at most `max_gap` frames
by linear interpolation between the neighbouring visible positions, within one period."""
import numpy as np


def fill_gaps(ball, periods, max_gap):
    """Returns (filled, source) with source 0 = observed, 2 = interpolated, -1 = still missing."""
    filled = ball.copy()
    visible = np.isfinite(ball).all(axis=1)
    source = np.where(visible, 0, -1)
    if max_gap <= 0:
        return filled, source
    for h in np.unique(periods[periods > 0]):
        idx = np.flatnonzero((periods == h) & visible)
        gaps = np.diff(idx) - 1
        for k in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
            a, b = idx[k], idx[k + 1]
            w = np.arange(1, b - a)[:, None] / (b - a)
            filled[a + 1:b] = ball[a] * (1 - w) + ball[b] * w
            source[a + 1:b] = 2
    return filled, source
