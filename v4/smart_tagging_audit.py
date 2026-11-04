"""Label-free reliability audit of Smart Tagging against the tracking. LOCAL MACHINE ONLY.

    OMP_NUM_THREADS=2 .venv/bin/python -m v4.smart_tagging_audit --data-dir data

Reads the tactical ATD and Smart Tagging XML; never reads any ground truth. Prints aggregates only
(per tag code: interval count, frames covered, agreement rates), never frames, arrays or tags, so
the output is safe to share. Purpose: learn what each code's Team tag means (team in possession or
defending team) and how reliable it is, before any fusion is designed.
"""
import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'starter-repo'))
FPS = 25


def parse_tags(path):
    out = []
    for inst in ET.parse(path).iter('instance'):
        tags = {lab.findtext('group'): lab.findtext('text') for lab in inst.findall('label')}
        out.append({'code': (inst.findtext('code') or '').strip().upper(),
                    'start': int(round(float(inst.findtext('start')) * FPS)),
                    'end': int(round(float(inst.findtext('end')) * FPS)),
                    'team': (tags.get('Team') or '').strip(), 'type': (tags.get('Type') or '').strip()})
    return out


def audit(tags, owner, visible, pred, periods):
    """Agreement of each code's Team tag with the nearest-player team and with v2, per club mapping."""
    clubs = sorted({t['team'] for t in tags if t['team'] and t['team'].upper() != 'N/A'})
    by_code = defaultdict(lambda: {'n': 0, 'frames': 0, 'vis': 0, 'owner_agree': [0, 0], 'pred_agree': [0, 0],
                                   'pred_dead': 0})
    for t in tags:
        a, b = max(t['start'], 0), min(t['end'], len(periods))
        if b <= a:
            continue
        rows = np.arange(a, b)[periods[a:b] > 0]
        s = by_code[t['code'] if t['code'] != 'SET PIECES' else f"SET PIECES / {t['type'] or '?'}"]
        s['n'] += 1
        s['frames'] += len(rows)
        if t['team'] not in clubs or not len(rows):
            continue
        k = clubs.index(t['team'])  # mapping m: clubs[0] -> side m, clubs[1] -> side 1 - m
        v = rows[visible[rows] & (owner[rows] >= 0)]
        s['vis'] += len(v)
        for m in (0, 1):
            side = k if m == 0 else 1 - k
            s['owner_agree'][m] += int(np.sum(owner[v] == side))
            s['pred_agree'][m] += int(np.sum(pred[rows] == 'HA'[side]))
        s['pred_dead'] += int(np.sum(pred[rows] == 'D'))
    report = {}
    for code, s in sorted(by_code.items()):
        report[code] = {'intervals': s['n'], 'frames': s['frames'], 'visible_owner_frames': s['vis'],
                        'owner_agree_mapping0': round(s['owner_agree'][0] / s['vis'], 3) if s['vis'] else None,
                        'owner_agree_mapping1': round(s['owner_agree'][1] / s['vis'], 3) if s['vis'] else None,
                        'v2_agree_mapping0': round(s['pred_agree'][0] / s['frames'], 3) if s['frames'] else None,
                        'v2_agree_mapping1': round(s['pred_agree'][1] / s['frames'], 3) if s['frames'] else None,
                        'v2_dead_share': round(s['pred_dead'] / s['frames'], 3) if s['frames'] else None}
    return clubs, report


def restart_alignment(tags, pred, periods):
    """For set pieces: does v2 show a dead-to-alive switch inside the tag window, and how far from its start?"""
    offsets, hits, total = [], 0, 0
    for t in tags:
        if t['code'] != 'SET PIECES':
            continue
        total += 1
        a, b = max(t['start'] - 125, 1), min(t['end'] + 125, len(pred))
        idx = np.arange(a, b)
        switch = idx[(pred[idx - 1] == 'D') & np.isin(pred[idx], ['H', 'A']) & (periods[idx] > 0)]
        if len(switch):
            hits += 1
            offsets.append(float((switch[np.argmin(np.abs(switch - t['start']))] - t['start']) / FPS))
    q = np.percentile(offsets, [10, 50, 90]).round(2).tolist() if offsets else None
    return {'set_pieces': total, 'with_v2_restart_within_5s_margin': hits, 'offset_s_p10_p50_p90': q}


POSSESSION_CODES = ('BUILD UP', 'PROGRESSION', 'ATTACKING TRANSITION', 'LONG BALL')  # EXPERIMENT-07


def phase_agreement(tags, pred, periods, club_side):
    hit = total = 0
    for t in tags:
        if t['code'] not in POSSESSION_CODES or t['team'] not in club_side:
            continue
        a, b = max(t['start'], 0), min(t['end'], len(periods))
        rows = np.arange(a, b)[periods[a:b] > 0]
        hit += int(np.sum(pred[rows] == club_side[t['team']]))
        total += len(rows)
    return round(hit / total, 4) if total else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    from games import asset_paths
    from possession import load_atd
    from v2 import methods as mt
    players, teams, ball, periods = load_atd(args.data_dir)
    obs = mt.observe(players, teams, ball, periods)
    dev = json.loads((ROOT / 'work/v2-evaluation/dev.json').read_text())
    model = pickle.loads((ROOT / 'work/v2-evaluation/model.pkl').read_bytes())  # project-trained
    q = mt.learned_probs(model, mt.learned_inputs(players, teams, ball, periods))
    pred = mt.blend(q, mt.prior50(obs, periods), periods, *dev['selected']['blend_learned'])
    from v4.infer import load_frozen, competition_inputs, predict
    m4, variant, setting = load_frozen()
    pred4 = predict(m4, *competition_inputs(args.data_dir), variant, setting)
    tags = parse_tags(asset_paths(data_dir=str(args.data_dir))['smart_tagging'])
    meta = ET.parse(asset_paths(data_dir=str(args.data_dir))['atd_meta']).getroot()
    team_names = [t.findtext('Name') or t.get('name') for t in meta.iter('Team')]
    clubs, report = audit(tags, obs['owner'], obs['visible'], pred, periods)
    _, report4 = audit(tags, obs['owner'], obs['visible'], pred4, periods)
    # Barcelona is ATD team 0 (H), established in the first audit run.
    club_side = {'Barcelona': 'H', 'Atletic Club': 'A'}
    g2, g4 = phase_agreement(tags, pred, periods, club_side), phase_agreement(tags, pred4, periods, club_side)
    from v6.infer import load_frozen as load6, competition_inputs as inputs6, predict as predict6
    pred6 = predict6(load6(), *inputs6(args.data_dir))
    g6 = phase_agreement(tags, pred6, periods, club_side)
    _, report6 = audit(tags, obs['owner'], obs['visible'], pred6, periods)
    verdict = None if g2 is None or g4 is None else ('corroborated' if g4 >= g2 - .02 else 'RED FLAG')
    print(json.dumps({'tag_clubs_sorted': clubs, 'atd_team_order_names': team_names,
                      'experiment07_phase_agreement': {'v2': g2, 'v4': g4, 'verdict': verdict},
                      'experiment10_v6_phase_agreement': {'v6': g6, 'v4': g4,
                                                          'verdict': 'ok' if g6 >= g4 - .02 else 'RED FLAG'},
                      'per_code_v6_dead_share': {k: v['v2_dead_share'] for k, v in report6.items()},
                      'restarts_v6': restart_alignment(tags, pred6, periods),
                      'per_code_v4_agree_correct_mapping': {k: v['v2_agree_mapping1'] for k, v in report4.items()},
                      'per_code_v2_agree_correct_mapping': {k: v['v2_agree_mapping1'] for k, v in report.items()},
                      'per_code_v4_dead_share': {k: v['v2_dead_share'] for k, v in report4.items()},
                      'restarts_v2': restart_alignment(tags, pred, periods),
                      'restarts_v4': restart_alignment(tags, pred4, periods)}, indent=1))
