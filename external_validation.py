"""External-game proxy comparison; never reads competition match data or labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from possession import features, baseline, decode, ROOT
sys.path.insert(0, str(ROOT / 'starter-repo' / 'evaluation'))
import possession_metrics as pm


def load_sample(game):
    folder = ROOT / 'external/metrica-sample/data' / f'Sample_Game_{game}'
    tracks = [pd.read_csv(folder / f'Sample_Game_{game}_RawTrackingData_{team}_Team.csv',
                          skiprows=2) for team in ['Home', 'Away']]
    if not tracks[0][['Frame', 'Period']].equals(tracks[1][['Frame', 'Period']]):
        raise ValueError('home and away tracking timelines differ')
    frames = tracks[0]['Frame'].to_numpy()
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError('external frame IDs must be one-based and contiguous')
    players = [df.iloc[:, 3:-2].to_numpy(float).reshape(len(df), -1, 2) for df in tracks]
    ball = tracks[0].iloc[:, -2:].to_numpy(float)
    if not np.allclose(ball, tracks[1].iloc[:, -2:].to_numpy(float), equal_nan=True):
        raise ValueError('home and away ball positions differ')
    teams = np.repeat([0, 1], [x.shape[1] for x in players])
    events = pd.read_csv(folder / f'Sample_Game_{game}_RawEventsData.csv')
    return np.concatenate(players, axis=1), teams, ball, tracks[0]['Period'].to_numpy(), events


def proxy_labels(events, periods):
    anchors = {}
    for _, event in events.iterrows():
        team = {'Home': 'H', 'Away': 'A'}.get(event['Team'])
        typ, sub = event['Type'], str(event['Subtype'])
        frame, label = None, None
        if team and typ in ['PASS', 'SET PIECE']:
            frame, label = int(event['Start Frame']) - 1, team
        elif team and typ == 'RECOVERY':
            frame, label = int(event['End Frame']) - 1, team
        elif typ == 'BALL OUT' or (typ == 'SHOT' and ('OUT' in sub or 'GOAL' in sub)):
            frame, label = int(event['End Frame']) - 1, 'D'
        elif typ == 'FAULT RECEIVED':
            frame, label = int(event['Start Frame']) - 1, 'D'
        if frame is not None:
            if not 0 <= frame < len(periods):
                raise ValueError('event anchor frame outside tracking bounds')
            if int(event['Period']) != periods[frame]:
                raise ValueError('event period disagrees with frame index')
            anchors.setdefault(frame, set()).add(label)
    out = np.full(len(periods), 'X', dtype='<U1')
    current, last_period = 'X', None
    for i, period in enumerate(periods):
        if period != last_period:
            current, last_period = 'X', period
        if i in anchors:
            current = next(iter(anchors[i])) if len(anchors[i]) == 1 else 'X'
        out[i] = current
    if np.sum(out != 'X') < len(out) / 2:
        raise ValueError('event-derived labels cover less than half the sample')
    return out


def mask_ball(ball):
    result = ball.copy()
    rng = np.random.default_rng(20261028)
    starts = np.arange(0, len(ball), 50)
    chosen = rng.choice(starts, round(.41 * len(starts)), replace=False)
    for start in chosen:
        result[start:start + 50] = np.nan
    return result


def measure(gt, pred, periods):
    # Score each half separately, so carrying abstentions cannot join the halves.
    reports = []
    for half in np.unique(periods):
        mask = periods == half
        r = pm.evaluate(pm.label_codes(gt[mask]), pm.label_codes(pred[mask]))
        reports.append(r)
    # Macro averaging over halves is an external experiment convention, not the competition board.
    return {'spellF1': float(np.mean([r['spell_f1_mean'] for r in reports])),
            'macroF1': float(np.mean([r['macro_f1'] for r in reports]))}


def run(game, penalties, missing_dead_cost=.5):
    players, teams, ball, periods, events = load_sample(game)
    gt = proxy_labels(events, periods)
    rows = []
    for condition in ['clean', 'masked']:
        observed = ball if condition == 'clean' else mask_ball(ball)
        costs, owner = features(players, teams, observed, periods, missing_dead_cost=missing_dead_cost)
        predictions = {'baseline50': baseline(owner, periods)}
        predictions.update({f'temporal{p:g}': decode(costs, periods, p) for p in penalties})
        for name, pred in predictions.items():
            row = {'game': game, 'condition': condition, 'method': name,
                   'graded_frames': int(np.sum(gt != 'X')), **measure(gt, pred, periods)}
            rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'holdout'])
    parser.add_argument('--revision', choices=['initial', 'neutral-missing'], default='initial')
    args = parser.parse_args()
    result_dir = ROOT / 'work/external-evaluation' / args.revision
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / (args.stage + '.json')
    if output.exists():
        raise SystemExit('Refusing to overwrite recorded experiment; inspect or amend explicitly.')
    if args.stage == 'dev':
        penalties = [5., 15., 40.] if args.revision == 'initial' else [.1, .5, 1.5, 5.]
        rows = run(1, penalties, 1.5 if args.revision == 'initial' else .5)
        candidates = sorted({r['method'] for r in rows if r['method'].startswith('temporal')},
                            key=lambda name: float(name[8:]))
        scores = {name: float(np.mean([r['spellF1'] for r in rows if r['method'] == name]))
                  for name in candidates}
        best = max(scores.values())
        selected = next(name for name in candidates if scores[name] >= best - .005)
        report = {'stage': 'development', 'proxy_warning': 'Event-derived labels, not challenge truth',
                  'rows': rows, 'selected_penalty': float(selected[8:]), 'scores': scores}
    else:
        dev = json.loads((result_dir / 'dev.json').read_text())
        penalty = dev['selected_penalty']
        rows = run(2, [penalty], 1.5 if args.revision == 'initial' else .5)
        pairs = {c: {r['method']: r for r in rows if r['condition'] == c}
                 for c in ['clean', 'masked']}
        candidate = f'temporal{penalty:g}'
        gains = {c: d[candidate]['spellF1'] - d['baseline50']['spellF1'] for c, d in pairs.items()}
        clean_macro_delta = pairs['clean'][candidate]['macroF1'] - pairs['clean']['baseline50']['macroF1']
        passed = all(x > 0 for x in gains.values()) and clean_macro_delta >= -.03
        report = {'stage': 'one-time holdout', 'proxy_warning': 'Event-derived labels, not challenge truth',
                  'rows': rows, 'spellF1_deltas': gains, 'clean_macroF1_delta': clean_macro_delta,
                  'provisional_gate_passed': passed, 'selected_penalty': penalty}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
