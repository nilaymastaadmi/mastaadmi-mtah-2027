"""Fresh external test of frozen candidate; never load challenge ground truth."""
import hashlib
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from possession import ROOT, baseline
from learned_possession import inputs, gate
from hybrid_possession import predict
from external_validation import proxy_labels, measure, pm
from verified_inference import check_freeze


def epts_layout(meta):
    score = meta.find('.//Score')
    ids = [score.get('idLocalTeam'), score.get('idVisitingTeam')]
    player_ids = [p.get('id') for p in meta.iter('Player')]
    player_team = {p.get('id'): p.get('teamId') for p in meta.iter('Player')}
    if len(set(ids)) != 2 or any(t not in ids for t in player_team.values()):
        raise ValueError('ambiguous team assignments')
    channels = {c.get('id'): (c.get('playerId'), c.get('channelId'))
                for c in meta.iter('PlayerChannel')}
    specs = []
    for spec in meta.iter('DataFormatSpecification'):
        groups = spec.findall('./SplitRegister')[0].findall('./SplitRegister')
        slots = []
        for group in groups:
            refs = [channels[c.get('playerChannelId')] for c in group.findall('PlayerChannelRef')]
            if len(refs) != 2 or refs[0][0] != refs[1][0] or [r[1] for r in refs] != ['x', 'y']:
                raise ValueError('unsupported coordinate channel order')
            slots.append(player_ids.index(refs[0][0]))
        if len(slots) != len(set(slots)):
            raise ValueError('duplicate player channel in format')
        specs.append((int(spec.get('startFrame')), int(spec.get('endFrame')), slots))
    specs.sort()
    if not specs or specs[0][0] != 1:
        raise ValueError('layout must start at frame one')
    for prior, current in zip(specs[:-1], specs[1:]):
        if prior[1] + 1 != current[0]:
            raise ValueError('format intervals must partition frame timeline')
    teams = np.array([ids.index(player_team[p]) for p in player_ids])
    return ids, specs, teams


def load_game3():
    folder = ROOT / 'external/metrica-sample/data/Sample_Game_3'
    meta = ET.parse(folder / 'Sample_Game_3_metadata.xml').getroot()
    ids, specs, teams = epts_layout(meta)
    n = specs[-1][1]
    players = np.full((n, len(teams), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    seen = np.zeros(n, bool)
    which = 0
    with (folder / 'Sample_Game_3_tracking.txt').open() as handle:
        for line in handle:
            frame, raw_players, raw_ball = line.strip().split(':')
            frame = int(frame)
            if not 1 <= frame <= n or seen[frame - 1]:
                raise ValueError('bad or duplicate frame')
            while which < len(specs) and frame > specs[which][1]:
                which += 1
            if which >= len(specs) or frame < specs[which][0]:
                raise ValueError('frame has no layout')
            coords = np.array([[float(v) for v in g.split(',')] for g in raw_players.split(';')])
            slots = specs[which][2]
            if coords.shape != (len(slots), 2):
                raise ValueError('tracking slots differ from XML layout')
            players[frame - 1, slots] = coords
            ball[frame - 1] = [float(v) for v in raw_ball.split(',')]
            seen[frame - 1] = True
    if not seen.all():
        raise ValueError('tracking frame missing')
    params = {p.findtext('Name'): p.findtext('Value') for p in
              meta.findall('.//ProviderGlobalParameters/ProviderParameter')}
    periods = np.zeros(n, int)
    for half, prefix in [(1, 'first_half'), (2, 'second_half')]:
        start, end = [int(params[prefix + suffix]) for suffix in ['_start', '_end']]
        periods[start - 1:end] = half
    if not (periods > 0).all():
        raise ValueError('unassigned external periods')
    events = json.loads((folder / 'Sample_Game_3_events.json').read_text())['data']
    rows = []
    for event in events:
        team_id = (event.get('team') or {}).get('id')
        if team_id is not None and team_id not in ids:
            raise ValueError('unknown event team')
        subtypes = event.get('subtypes')
        if isinstance(subtypes, list):
            subtype = '-'.join(s['name'] for s in subtypes)
        else:
            subtype = (subtypes or {}).get('name', '')
        rows.append({'Team': dict(zip(ids, ['Home', 'Away'])).get(team_id),
                     'Type': event['type']['name'], 'Subtype': subtype,
                     'Period': event['period'], 'Start Frame': event['start']['frame'],
                     'End Frame': event['end']['frame']})
    return players, teams, ball, periods, pd.DataFrame(rows)


def stress(ball, block):
    observed = ball.copy()
    starts = np.arange(0, len(ball), block)
    chosen = np.random.default_rng(20261029).choice(starts, round(.41 * len(starts)), replace=False)
    for start in chosen:
        observed[start:start + block] = np.nan
    return observed


def secondary(gt, pred, periods, events):
    known = gt != 'X'
    boundary = np.zeros(len(gt), bool)
    for i in np.flatnonzero((gt[1:] != gt[:-1]) | (periods[1:] != periods[:-1])) + 1:
        boundary[max(0, i - 25):min(len(gt), i + 26)] = True
    interior = known & ~boundary
    pass_mask, pass_target = np.zeros(len(gt), bool), np.full(len(gt), 'X', dtype='<U1')
    for _, event in events[events.Type == 'PASS'].iterrows():
        a, b = int(event['Start Frame']) - 1, int(event['End Frame'])
        if a < 0 or b > len(gt) or b < a:
            raise ValueError('pass interval out of bounds')
        pass_mask[a:b] = True
        pass_target[a:b] = {'Home': 'H', 'Away': 'A'}[event['Team']]
    anchor_correct = []
    for _, event in events[events.Type.isin(['PASS', 'SET PIECE', 'RECOVERY'])].iterrows():
        i = int(event['End Frame'] if event['Type'] == 'RECOVERY' else event['Start Frame']) - 1
        anchor_correct.append(pred[i] == {'Home': 'H', 'Away': 'A'}[event['Team']])
    return {'interior_accuracy': float(np.mean(pred[interior] == gt[interior])),
            'pass_interval_accuracy': float(np.mean(pred[pass_mask] == pass_target[pass_mask])),
            'control_anchor_accuracy': float(np.mean(anchor_correct)),
            'n_control_anchors': len(anchor_correct)}


def frozen_model():
    folder = ROOT / 'work/hybrid-evaluation'
    dev, held = [json.loads((folder / f'{name}.json').read_text()) for name in ['dev', 'holdout']]
    upstream = json.loads((ROOT / 'work/learned-evaluation/dev.json').read_text())
    path = ROOT / 'work/learned-evaluation/model.pkl'
    model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in dev['source_hashes']}
    check_freeze(dev, held, upstream, model_hash, hashes)
    with path.open('rb') as f:
        model = pickle.load(f)
    return model, dev['selected'], model_hash


if __name__ == '__main__':
    folder = ROOT / 'work/first-assessment'
    model, (weight, penalty), model_hash = frozen_model()
    pinned = json.loads((folder / 'pinned.json').read_text())
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in pinned.items()):
        raise SystemExit('Pinned object changed')
    with (folder / 'fresh-test.started').open('x') as f:
        json.dump({'model_sha256': model_hash, 'weight': weight, 'penalty': penalty,
                   'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, f)
    players, teams, ball, periods, events = load_game3()
    gt = proxy_labels(events, periods)
    rows = []
    for name, block in [('natural', None), ('block10', 10), ('block50', 50), ('block250', 250)]:
        observed = ball if block is None else stress(ball, block)
        x, owner = inputs(players, teams, observed, periods)
        methods = {'baseline_raw': baseline(owner, periods, 0),
                   'baseline50': baseline(owner, periods, 50),
                   'baseline125': baseline(owner, periods, 125),
                   'hybrid': predict(model, x, owner, periods, weight, penalty),
                   'no_learning': predict(model, x, owner, periods, 0., penalty)}
        for method, pred in methods.items():
            half_rows = []
            for half in [1, 2]:
                mask = periods == half
                result = pm.evaluate(pm.label_codes(gt[mask]), pm.label_codes(pred[mask]))
                half_rows.append({'half': half, 'spellF1': result['spell_f1_mean'],
                                  'macroF1': result['macro_f1'], 'gt_spells': result['n_gt_spells'],
                                  'pred_spells': result['n_pred_spells'],
                                  'D_f1': result['per_class']['D']['f1']})
            rows.append({'condition': name, 'method': method, **measure(gt, pred, periods),
                         **secondary(gt, pred, periods, events), 'halves': half_rows,
                         'missing_ball_fraction': float(np.mean(~np.isfinite(observed).all(axis=1)))})
    pairs = {c: {r['method']: r for r in rows if r['condition'] == c}
             for c in ['natural', 'block10', 'block50', 'block250']}
    deltas = {c: {m: d['hybrid']['spellF1'] - d[m]['spellF1'] for m in
                  ['baseline50', 'baseline125', 'no_learning']} for c, d in pairs.items()}
    macro_delta = pairs['natural']['hybrid']['macroF1'] - pairs['natural']['baseline50']['macroF1']
    passed = deltas['natural']['baseline50'] > 0 and deltas['block50']['baseline50'] > 0 and macro_delta >= -.03
    report = {'unit': 'one fresh external match; four correlated conditions', 'rows': rows,
              'spellF1_deltas': deltas, 'natural_macro_delta': macro_delta,
              'predeclared_gate_passed': passed, 'model_sha256': model_hash,
              'candidate_settings': [weight, penalty], 'proxy_graded_frames': int(np.sum(gt != 'X'))}
    (folder / 'fresh-test.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Fresh external test saved; no model settings changed.')
