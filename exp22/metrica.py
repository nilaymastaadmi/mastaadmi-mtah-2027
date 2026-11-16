"""EXPERIMENT-22: Metrica Sample Games 1-3 as training views (work/EXPERIMENT-22.md, binding truth rule).

    .venv/bin/python -m exp22.metrica check      # instrument checks, aggregates only; work/exp22/metrica-instrument.json

load(game) returns the arrays the degradation and feature code expect: players (N x slots x 2, [0, 1]), teams (0 home,
1 away), ball (N x 2), z (zeros: height unknown), periods (1, 2; 0 outside), pitch (105, 68), labels (H/A/D/X).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / 'external/metrica-sample/data'
GAMES = ('1', '2', '3')
PITCH = np.array([105., 68.])
IN_PLAY = {'PASS', 'CARRY', 'RECOVERY', 'SHOT', 'SET PIECE', 'BALL LOST', 'BALL OUT', 'CHALLENGE'}
ANCHOR = {'PASS', 'CARRY', 'RECOVERY', 'SHOT', 'SET PIECE'}
KEEP_ONLY = {'BALL LOST', 'BALL OUT'}


# ---------- events, one normalised table for all three games ----------

def events(game):
    """DataFrame: team (H/A/None), type, subtype (upper string), period, start, end (0-based frame indices)."""
    folder = DATA / f'Sample_Game_{game}'
    if game in ('1', '2'):
        e = pd.read_csv(folder / f'Sample_Game_{game}_RawEventsData.csv')
        out = pd.DataFrame({'team': e['Team'].map({'Home': 'H', 'Away': 'A'}), 'type': e['Type'].str.upper(),
                            'subtype': e['Subtype'].fillna('').astype(str).str.upper(), 'period': e['Period'],
                            'start': e['Start Frame'] - 1, 'end': e['End Frame'] - 1})
    else:
        meta = ET.parse(folder / 'Sample_Game_3_metadata.xml').getroot()
        home = next(meta.iter('Score')).get('idLocalTeam')
        rows = []
        for ev in json.loads((folder / 'Sample_Game_3_events.json').read_text())['data']:
            sub = ev.get('subtypes')
            if isinstance(sub, list):
                sub = '-'.join(s['name'] for s in sub)
            elif isinstance(sub, dict):
                sub = sub['name']
            team = ev.get('team') or {}
            rows.append({'team': None if not team.get('id') else ('H' if team['id'] == home else 'A'),
                         'type': ev['type']['name'].upper(), 'subtype': (sub or '').upper(),
                         'period': ev['period'], 'start': ev['start']['frame'] - 1, 'end': ev['end']['frame'] - 1})
        out = pd.DataFrame(rows)
    out['end'] = np.maximum(out['end'], out['start'])           # Game 1 kick-off has End Frame 0
    out = out.sort_values(['start', 'end'], kind='stable').reset_index(drop=True)
    return out


def stoppage_frame(ev):
    """Frame where a stoppage event makes the ball dead, or None (registration rule 1b)."""
    t, s = ev['type'], ev['subtype']
    if t == 'BALL OUT':
        return ev['end']
    if t == 'SHOT' and ('OUT' in s or 'GOAL' in s):
        return ev['end']
    if t == 'FAULT RECEIVED':
        return ev['start']
    if t == 'BALL LOST' and any(k in s for k in ('OFFSIDE', 'HAND BALL', 'END HALF')):
        return ev['end']
    return None


def truth(ev_all, periods):
    """Frame labels under the registered rule. periods: 1 or 2 per frame, 0 outside play."""
    n = len(periods)
    out = np.full(n, 'X', dtype='<U1')
    for half in (1, 2):
        rows = np.flatnonzero(periods == half)
        if len(rows) == 0:
            continue
        lo, hi = rows[0], rows[-1] + 1
        ev = ev_all[ev_all['period'] == half].reset_index(drop=True)
        ev = ev[(ev['start'] >= lo) & (ev['start'] < hi)].reset_index(drop=True)
        if ev.empty:
            continue
        lab = np.full(hi - lo, 'X', dtype='<U1')
        # rule 2: controlled possession from anchors, carried forward
        state = 'X'
        anchors = []
        for _, e in ev.iterrows():
            team = e['team'] if e['team'] in ('H', 'A') else None
            if team is None:
                continue
            if e['type'] in ANCHOR:
                state = team
            elif e['type'] in KEEP_ONLY and state == team:
                pass
            else:
                continue
            anchors.append((int(e['start']), state))
        cur, j = 'X', 0
        first = int(ev['start'].min())
        for i in range(first, hi):
            while j < len(anchors) and anchors[j][0] <= i:
                cur = anchors[j][1]
                j += 1
            lab[i - lo] = cur
        # rule 1: dead spells overwrite
        dead = np.zeros(hi - lo, bool)
        inplay = ev[ev['type'].isin(IN_PLAY)].reset_index(drop=True)
        for k, e in inplay.iterrows():
            if e['type'] != 'SET PIECE':
                continue
            prev = inplay[(inplay.index < k) & (inplay['start'] <= e['start'])]
            if len(prev):
                a = int(prev['end'].iloc[-1])
                b = int(e['start'])
                if a < b:
                    dead[max(a, lo) - lo:b - lo] = True
        for k, e in ev.iterrows():
            f = stoppage_frame(e)
            if f is None:
                continue
            later = inplay[(inplay['start'] > f) | ((inplay['start'] == f) & (inplay['type'] == 'SET PIECE'))]
            later = later[later['start'] >= f]
            b = int(later['start'].iloc[0]) if len(later) else hi
            dead[max(f, lo) - lo:min(b, hi) - lo] = True
        lab[dead] = 'D'
        # rule 3: frames after the last event are X unless the last event is a stoppage
        last = ev.iloc[-1]
        if stoppage_frame(last) is None:
            tail = int(max(ev['end'].max(), ev['start'].max())) + 1
            lab[max(tail, lo) - lo:] = 'X'
        lab[:max(first, lo) - lo] = 'X'
        out[lo:hi] = lab
    return out


# ---------- tracking ----------

def _track_12(game):
    from external_validation import load_sample
    players, teams, ball, periods, _ = load_sample(int(game))
    return players, teams, ball, np.asarray(periods, int)


def _track_3():
    folder = DATA / 'Sample_Game_3'
    meta = ET.parse(folder / 'Sample_Game_3_metadata.xml').getroot()
    home = next(meta.iter('Score')).get('idLocalTeam')
    ids = [p.get('id') for p in meta.iter('Player')]
    team_of = {p.get('id'): (0 if p.get('teamId') == home else 1) for p in meta.iter('Player')}
    order = sorted(ids, key=lambda p: (team_of[p], ids.index(p)))
    slot = {p: i for i, p in enumerate(order)}
    chan = {c.get('id'): c.get('playerId') for c in meta.iter('PlayerChannel')}
    specs = []
    for s in meta.iter('DataFormatSpecification'):
        refs = [chan[r.get('playerChannelId')] for r in s.iter('PlayerChannelRef')
                if r.get('playerChannelId').endswith('_x')]
        specs.append((int(s.get('startFrame')), int(s.get('endFrame')), [slot[p] for p in refs]))
    par = {p.find('Name').text: p.find('Value').text for p in meta.iter('ProviderParameter')
           if p.find('Name') is not None}
    halves = [(int(par['first_half_start']), int(par['first_half_end'])),
              (int(par['second_half_start']), int(par['second_half_end']))]
    n = halves[1][1]
    players = np.full((n, len(order), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(folder / 'Sample_Game_3_tracking.txt') as fh:
        for line in fh:
            fr, pl, bl = line.strip().split(':')
            i = int(fr)
            spec = next(s for s in specs if s[0] <= i <= s[1])
            vals = np.array([g.split(',') for g in pl.split(';')], float)
            if len(vals) != len(spec[2]):
                raise ValueError(f'frame {i}: {len(vals)} players, spec has {len(spec[2])}')
            players[i - 1, spec[2]] = vals
            ball[i - 1] = np.array(bl.split(',')[:2], float)
    periods = np.zeros(n, int)
    for h, (a, b) in enumerate(halves, start=1):
        periods[a - 1:b] = h
    teams = np.array([team_of[p] for p in order])
    return players, teams, ball, periods


def load(game):
    players, teams, ball, periods = _track_3() if game == '3' else _track_12(game)
    labels = truth(events(game), periods)
    return {'players': players, 'teams': teams, 'ball': ball, 'periods': periods, 'labels': labels,
            'z': np.zeros(len(ball)), 'pitch': PITCH.copy()}


def check():
    from v2 import experiment as ex2
    rows = {}
    for g in GAMES:
        a = load(g)
        inp = a['periods'] > 0
        lab = a['labels'][inp]
        graded = lab[lab != 'X']
        runs = 0
        for half in (1, 2):
            l = a['labels'][a['periods'] == half]
            l = l[l != 'X']
            runs += int(np.sum(np.isin(l, ['H', 'A']) & np.r_[True, l[1:] != l[:-1]]))
        minutes = len(graded) / 25 / 60
        r = {'frames': int(inp.sum()), 'graded_minutes': round(minutes, 1), 'mapping': ex2.mapping_check(a),
             'd_share': float(np.mean(graded == 'D')), 'x_share': float(np.mean(lab == 'X')),
             'possessions': runs, 'possessions_per_min': runs / minutes,
             'ball_share': float(np.mean(np.isfinite(a['ball'][inp]).all(1))),
             'h_share': float(np.mean(graded == 'H'))}
        r['ok'] = (r['mapping'] > .5 and .15 <= r['d_share'] <= .65 and 1.5 <= r['possessions_per_min'] <= 6
                   and r['x_share'] < .10)
        rows[g] = r
    out = ROOT / 'work/exp22/metrica-instrument.json'
    out.write_text(json.dumps(rows, indent=1) + '\n')
    print(json.dumps(rows, indent=1))
    return rows


if __name__ == '__main__':
    check()
