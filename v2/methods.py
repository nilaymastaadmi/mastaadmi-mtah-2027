"""v2 comparators. v1 modules are imported read-only; none of their files are modified."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import features, baseline  # noqa: E402  (frozen v1)
from learned_possession import inputs  # noqa: E402  (frozen v1 feature set)

STATES = 'HAD'


def viterbi(costs, periods, penalty):
    """Same objective as v1 possession.decode (uniform switch penalty), in plain Python floats.

    Equal-cost ties may resolve differently from v1; tests check agreement on continuous costs.
    """
    costs, periods = np.asarray(costs, float), np.asarray(periods)
    if costs.ndim != 2 or costs.shape[1] != 3 or len(periods) != len(costs):
        raise ValueError('costs must be N by 3 and match periods')
    if not np.isfinite(costs).all() or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('invalid costs or penalty')
    out = np.full(len(costs), 'X', dtype='<U1')
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] <= 0:
            continue
        c = costs[start:end].tolist()
        n = end - start
        back = bytearray(3 * n)
        s0, s1, s2 = c[0]
        for i in range(1, n):
            if s0 <= s1 and s0 <= s2:
                m, arg = s0, 0
            elif s1 <= s2:
                m, arg = s1, 1
            else:
                m, arg = s2, 2
            t = m + penalty
            r0, r1, r2 = c[i]
            k = 3 * i
            if s0 <= t:
                s0 = s0 + r0
                back[k] = 0
            else:
                s0 = t + r0
                back[k] = arg
            if s1 <= t:
                s1 = s1 + r1
                back[k + 1] = 1
            else:
                s1 = t + r1
                back[k + 1] = arg
            if s2 <= t:
                s2 = s2 + r2
                back[k + 2] = 2
            else:
                s2 = t + r2
                back[k + 2] = arg
        final = [s0, s1, s2]
        state = final.index(min(final))
        path = np.empty(n, np.int8)
        for i in range(n - 1, -1, -1):
            path[i] = state
            state = back[3 * i + state]
        out[start:end] = np.array(list(STATES))[path]
    return out


def forward_fill(labels, periods):
    """Explicitly apply the scorer's X fill within each half (helps macroF1, spellF1 unchanged)."""
    out = labels.copy()
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] <= 0:
            continue
        window = out[start:end]
        known = np.flatnonzero(window != 'X')
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != 'X', np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:end] = window[carry]
    return out


def observe(players, teams, ball, periods):
    """Shared per-frame observations from frozen v1 code."""
    costs, owner = features(players, teams, ball, periods)
    visible = np.isfinite(ball).all(axis=1)
    outside = visible & ((ball < 0).any(axis=1) | (ball > 1).any(axis=1))
    return {'costs': costs, 'owner': owner, 'visible': visible, 'outside': outside}


def majority(obs, periods, window):
    """Official starter baseline family: missing ball votes D."""
    return baseline(obs['owner'], periods, window)


def hold_majority(obs, periods, window):
    """Missing ball abstains (carried forward); outside-pitch ball votes D; majority over window."""
    votes = np.full(len(periods), -1)
    votes[obs['owner'] >= 0] = obs['owner'][obs['owner'] >= 0]
    votes[obs['outside']] = 2
    out = np.full(len(periods), 'X', dtype='<U1')
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    for start, end in zip(edges[:-1], edges[1:]):
        if periods[start] <= 0:
            continue
        v = votes[start:end]
        cum = [np.r_[0, np.cumsum(v == k)] for k in range(3)]
        idx = np.arange(end - start)
        lo, hi = np.maximum(0, idx - window), np.minimum(end - start, idx + window + 1)
        counts = np.column_stack([c[hi] - c[lo] for c in cum])
        seen = counts.sum(axis=1) > 0
        label = np.array(list(STATES))[np.argmax(counts, axis=1)]
        out[start:end][seen] = label[seen]
    return forward_fill(out, periods)


def hmm_hand(obs, periods, penalty):
    """Frozen v1 hand-designed emission costs with a tuned switch penalty."""
    return viterbi(obs['costs'], periods, penalty)


def prior50(obs, periods):
    reference = baseline(obs['owner'], periods, 50)
    prior = np.full((len(periods), 3), .01)
    for k, label in enumerate(STATES):
        prior[reference == label, k] = .98
    return prior


def blend(q, prior, periods, weight, penalty):
    """v1 hybrid form with a generic evidence source q (learned, uniform or misaligned)."""
    mixture = weight * q + (1 - weight) * prior
    return viterbi(-np.log(np.clip(mixture, 1e-6, 1.)) / 25., periods, penalty)


def uniform_evidence(n):
    return np.full((n, 3), 1 / 3)


def misaligned(q, periods):
    """Circularly shift evidence by half of each half: same marginals and autocorrelation, wrong time."""
    out = q.copy()
    for half in np.unique(periods[periods > 0]):
        rows = np.flatnonzero(periods == half)
        out[rows] = np.roll(q[rows], len(rows) // 2, axis=0)
    return out


def learned_probs(model, x):
    probs = model.predict_proba(x)
    return probs[:, [list(model.classes_).index(c) for c in STATES]]


def learned_inputs(players, teams, ball, periods):
    return inputs(players, teams, ball, periods)[0]
