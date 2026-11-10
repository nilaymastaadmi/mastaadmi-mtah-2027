"""Change-event targets (numpy only, so the torch venv can import them)."""
import numpy as np


def change_events(labels):
    """Frame t where the label goes directly H->A or A->H (restarts after D are not change events)."""
    labels = np.asarray(labels)
    ev = np.zeros(len(labels), bool)
    ev[1:] = np.isin(labels[1:], ['H', 'A']) & np.isin(labels[:-1], ['H', 'A']) & (labels[1:] != labels[:-1])
    return ev


def near(events, k, periods):
    """True where an event lies within +-k frames in the same half."""
    out = np.zeros(len(events), bool)
    periods = np.asarray(periods)
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for a, b in zip(edges[:-1], edges[1:]):
        e = events[a:b].astype(int)
        c = np.r_[0, np.cumsum(e)]
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - k), np.minimum(b - a, i + k + 1)
        out[a:b] = (c[hi] - c[lo]) > 0
    return out
