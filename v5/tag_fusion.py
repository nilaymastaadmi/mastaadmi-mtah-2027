"""EXPERIMENT-09: Smart Tagging team fusion calibrated on degraded SkillCorner phase tags.

    OMP_NUM_THREADS=1 .venv/bin/python -m v5.tag_fusion

Writes work/v5-tagfusion/{rows.jsonl,selection.json}.
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import experiment as ex2, methods as mt  # noqa: E402
from v5.labels import relabel  # noqa: E402

OOF = ROOT / 'external/oof/raw'
META = ROOT / 'external/skillcorner/data/matches'
OUT = ROOT / 'work/v5-tagfusion'
TYPES = ('build_up', 'create', 'direct', 'quick_break', 'transition')
EPS = (.05, .10, .20)
LAMBDAS = (0., .25, .5, 1., 2., 4.)
CONVENTIONS = ('raw', 'controlled1', 'controlled2', 'controlled3')
W, PENALTY = 1., .5  # frozen v4 decoder


def fuse_costs(q, prior, tags, lam, w=W):
    """tags: array of 'H'/'A'/'' per frame. Adds lam/25 to the other team's cost inside tags."""
    costs = -np.log(np.clip(w * q + (1 - w) * prior, 1e-6, 1.)) / 25.
    if lam:
        costs[tags == 'H', 1] += lam / 25.
        costs[tags == 'A', 0] += lam / 25.
    return costs


def analog_tags(match, periods, eps):
    """Degraded SkillCorner possession-phase tags on the 25 fps full-match timeline."""
    info = json.loads((META / str(match) / f'{match}_match.json').read_text())
    home = info['home_team']['id']
    ph = pd.read_csv(META / str(match) / f'{match}_phases_of_play.csv')
    rng = np.random.default_rng(int(hashlib.sha256(f'{match}|{eps}|20261103'.encode()).hexdigest()[:16], 16))
    out = np.full(len(periods), '', dtype='<U1')
    # 25 fps index of each source 10 fps frame, per half, following v3.skillcorner.load layout.
    halves = sorted(info['match_periods'], key=lambda p: p['period'])[:2]
    offset = 0
    for half, p in enumerate(halves, start=1):
        start, end = int(p['start_frame']), int(p['end_frame'])
        n_out = int(np.floor((end - start) * 25 / 10)) + 1
        sel = ph[(ph.team_in_possession_phase_type.isin(TYPES)) & (ph.frame_end >= start) & (ph.frame_start <= end)]
        for _, r in sel.iterrows():
            team = 'H' if r.team_in_possession_id == home else 'A'
            if rng.random() < eps:
                team = 'A' if team == 'H' else 'H'
            a = int(round((max(r.frame_start, start) - start) * 2.5 + rng.normal(0, 25)))
            b = int(round((min(r.frame_end, end) - start) * 2.5 + rng.normal(0, 25)))
            a, b = max(a, 0), min(b, n_out - 1)
            if b > a:
                out[offset + a:offset + b + 1] = team
        offset += n_out + 1  # separator frame between halves
    out[periods <= 0] = ''
    return out


def view(match):
    d = np.load(OOF / f'sc-{match}-real.npz')
    q, periods, labels = d['q'], d['periods'], d['labels']
    prior = mt.prior50({'owner': d['owner']}, periods)
    truths = {c: relabel(labels, periods, c) for c in CONVENTIONS}
    rows = []
    for eps in EPS:
        tags = analog_tags(match, periods, eps)
        for lam in LAMBDAS:
            pred = mt.viterbi(fuse_costs(q, prior, tags, lam), periods, PENALTY)
            s = {c: ex2.score(truths[c], pred) for c in CONVENTIONS}
            rows.append({'match': match, 'eps': eps, 'lam': lam, 'tag_coverage': float(np.mean(tags[periods > 0] != '')),
                         **{f'spell_{c}': s[c]['spellF1'] for c in CONVENTIONS}, 'macro_raw': s['raw']['macroF1']})
    return rows


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    matches = sorted(int(p.name.split('-')[1]) for p in OOF.glob('sc-*-real.npz'))
    with ProcessPoolExecutor(4) as pool:
        rows = [r for chunk in pool.map(view, matches) for r in chunk]
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    conv = lambda r: float(np.mean([r[f'spell_{c}'] for c in CONVENTIONS]))
    def per(eps, lam):
        return {r['match']: r for r in rows if r['eps'] == eps and r['lam'] == lam}
    def primary(eps, lam):
        return float(np.mean([conv(r) for r in per(eps, lam).values()]))
    best = max(primary(.10, l) for l in LAMBDAS)
    lam = next(l for l in LAMBDAS if primary(.10, l) >= best - .005)
    d10 = [conv(per(.10, lam)[m]) - conv(per(.10, 0.)[m]) for m in matches]
    d20 = [conv(per(.20, lam)[m]) - conv(per(.20, 0.)[m]) for m in matches]
    macro = float(np.mean([per(.10, lam)[m]['macro_raw'] - per(.10, 0.)[m]['macro_raw'] for m in matches]))
    checks = {'primary_delta_ge_.02': float(np.mean(d10)) >= .02, 'wins_ge_14of20': int(np.sum(np.array(d10) > 0)) >= 14,
              'eps20_mean_ge_0': float(np.mean(d20)) >= 0, 'macro_ge_-.02': macro >= -.02}
    table = {f'eps{e}': {f'lam{l}': round(primary(e, l), 4) for l in LAMBDAS} for e in EPS}
    report = {'selected_lambda': lam, 'primary_by_eps_lambda': table, 'deltas_eps10': d10, 'deltas_eps20': d20,
              'macro_delta_eps10': macro, 'checks': checks, 'passed': all(checks.values()),
              'tag_coverage_mean': float(np.mean([r['tag_coverage'] for r in rows]))}
    (OUT / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ['selected_lambda', 'primary_by_eps_lambda', 'checks', 'passed', 'tag_coverage_mean']}, indent=1))
