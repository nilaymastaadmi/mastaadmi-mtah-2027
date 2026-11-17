"""H/A/D truth for PFF tracking frames from PFF's own game events (EXPERIMENT-19 label rule, fixed before any score).

PFF embeds the current game event in every tracking frame it overlaps (`game_event`, Tracking spec v2.2 section 5).
Distinct events are collected by `game_event_id` with their own `start_frame`. Walking the frames in time order,
each event changes the state at its start frame:

- OTB (a player on the ball) and the kick-offs (FIRSTKICKOFF, SECONDKICKOFF): the team in possession is the
  sequence-level `home_ball` (H if true, A if false). PFF defines a sequence as an uninterrupted possession by one
  team, so a touch by the other team that does not end the sequence does not flip the label. If `home_ball` is
  missing, `home_team` (the team of the player on the ball) is used; if both are missing the state is unchanged.
- OUT (ball out of play: touch, whistle, goal) and END: dead ball, D, until the next OTB or kick-off.
- Before the first state event of a period: D.
- Frames inside a VID (video missing) event, from its start to its end frame: X (not graded).
- All other game events (SUB, ON, OFF, G, FOUL) change nothing.
Ties at one start frame are applied in game_event_id order. Events are read from every record, including the
duplicated frames the loader drops. An event belongs to the half of the first frame that carries it; if its start
frame precedes that half's first tracked frame (a kick-off just before tracking starts), it applies from the first
frame (counted in `events_started_before_half`). Review fixes of 11 November, before any score.
"""
import numpy as np

TEAM_EVENTS = {'OTB', 'FIRSTKICKOFF', 'SECONDKICKOFF'}
DEAD_EVENTS = {'OUT', 'END'}


def events_of(records):
    out = {}
    for r in records:
        g = r.get('game_event')
        gid = r.get('game_event_id')
        if not g or gid is None:
            continue
        gid = int(gid)
        if gid not in out:
            out[gid] = {'type': g.get('game_event_type'), 'home_ball': g.get('home_ball'),
                        'home_team': g.get('home_team'), 'start': g.get('start_frame'), 'end': g.get('end_frame'),
                        'first_seen': int(r['frameNum']), 'period': r.get('period')}
    for e in out.values():
        if e['start'] is None:
            e['start'] = e['first_seen']
        if e['end'] is None:
            e['end'] = e['start']
    return out


def native_labels(records, extra=()):
    """records sorted by frameNum (periods 1 and 2); extra: dropped duplicate records (events only).
    Returns (labels per record, stats)."""
    frames = np.array([int(r['frameNum']) for r in records])
    periods = np.array([r['period'] for r in records])
    events = events_of(sorted(list(records) + list(extra), key=lambda r: int(r['frameNum'])))
    first = {h: int(frames[periods == h].min()) for h in (1, 2) if (periods == h).any()}
    last = {h: int(frames[periods == h].max()) for h in (1, 2) if (periods == h).any()}
    stats = {'events_otb': 0, 'events_out': 0, 'events_vid': 0, 'otb_home_ball_missing': 0,
             'otb_team_missing': 0, 'otb_home_ball_differs_from_player_team': 0, 'events_started_before_half': 0,
             'events_outside_halves': 0}
    changes = []
    for gid, e in sorted(events.items(), key=lambda kv: (int(kv[1]['start']), kv[0])):
        t = e['type']
        h = e['period']
        if h not in first:
            stats['events_outside_halves'] += 1
            continue
        start = int(e['start'])
        if start < first[h]:
            stats['events_started_before_half'] += t in TEAM_EVENTS | DEAD_EVENTS
            start = first[h]
        if start > last[h]:
            stats['events_outside_halves'] += 1
            continue
        if t in TEAM_EVENTS:
            stats['events_otb'] += 1
            hb = e['home_ball'] if e['home_ball'] is not None else e['home_team']
            if e['home_ball'] is None:
                stats['otb_home_ball_missing'] += 1
            if e['home_ball'] is not None and e['home_team'] is not None and bool(e['home_ball']) != bool(e['home_team']):
                stats['otb_home_ball_differs_from_player_team'] += 1
            if hb is None:
                stats['otb_team_missing'] += 1
                continue
            changes.append((start, 'H' if hb else 'A'))
        elif t in DEAD_EVENTS:
            stats['events_out'] += t == 'OUT'
            changes.append((start, 'D'))
    labels = np.full(len(records), 'D', dtype='<U1')
    for half in (1, 2):
        idx = np.flatnonzero(periods == half)
        if not len(idx):
            continue
        f = frames[idx]
        state, k = 'D', 0
        ch = [c for c in changes if f[0] <= c[0] <= f[-1]]
        for i, fr in zip(idx, f):
            while k < len(ch) and ch[k][0] <= fr:
                state = ch[k][1]
                k += 1
            labels[i] = state
    for e in events.values():
        if e['type'] == 'VID':
            stats['events_vid'] += 1
            labels[(frames >= int(e['start'])) & (frames <= int(e['end']))] = 'X'
    stats['native_share'] = {k: float(np.mean(labels == k)) for k in 'HADX'}
    return labels, stats
