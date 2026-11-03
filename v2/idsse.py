"""Load public IDSSE/DFL matches (CC BY 4.0) into v1-compatible arrays plus tracking-derived labels.

Source: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data 12:195, figshare doi 10.6084/m9.figshare.28196177.v1.
Raw files live only in ignored external/idsse/. Labels come from the provider's per-frame
BallStatus (0 dead, 1 alive) and BallPossession (1 home, 2 guest). They never enter features.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'external/idsse'
COMPETITION = {'J03WMX': '000001', 'J03WN1': '000001', 'J03WOH': '000002', 'J03WOY': '000002',
               'J03WPY': '000002', 'J03WQQ': '000002', 'J03WR9': '000002'}
# Fixed before any outcome by numpy.random.default_rng(20261102).permutation(sorted(ids)):
# the first three are development, the rest are the one-time holdout (EXPERIMENT-04).
DEVELOPMENT = ('J03WOH', 'J03WOY', 'J03WPY')
HOLDOUT = ('J03WMX', 'J03WN1', 'J03WQQ', 'J03WR9')
SECTIONS = ('firstHalf', 'secondHalf')


def paths(match, raw=RAW):
    comp = COMPETITION[match]
    stem = f'DFL-COM-{comp}_DFL-MAT-{match}.xml'
    return raw / f'DFL_02_01_matchinformation_{stem}', raw / f'DFL_04_03_positions_raw_observed_{stem}'


def parse_info(path):
    root = ET.parse(path).getroot()
    env = root.find('.//Environment')
    pitch = (float(env.get('PitchX')), float(env.get('PitchY')))
    roles = {t.get('Role'): t.get('TeamId') for t in root.iter('Team')}
    if set(roles) != {'home', 'guest'} or roles['home'] == roles['guest']:
        raise ValueError('match information must name one home and one guest team')
    return pitch, roles['home'], roles['guest']


def parse_positions(path, home, guest):
    """Stream the large XML; return {section: {'ball': arrays, 'players': {(team, person): arrays}}}."""
    out = {s: {'ball': None, 'players': {}} for s in SECTIONS}
    current, rows = None, []
    for event, elem in ET.iterparse(path, events=('start', 'end')):
        if event == 'start' and elem.tag == 'FrameSet':
            section, team = elem.get('GameSection'), elem.get('TeamId')
            if section not in SECTIONS:
                raise ValueError(f'unexpected game section {section}')
            current, rows = (section, team, elem.get('PersonId')), []
        elif event == 'end' and elem.tag == 'Frame' and current is not None:
            a = elem.attrib
            if current[1] == 'BALL':
                rows.append((int(a['N']), float(a['X']), float(a['Y']), float(a.get('Z', 'nan')),
                             int(a.get('BallPossession', 0)), int(a.get('BallStatus', -1))))
            else:
                rows.append((int(a['N']), float(a['X']), float(a['Y'])))
            elem.clear()
        elif event == 'end' and elem.tag == 'FrameSet':
            section, team, person = current
            data = np.array(rows, dtype=float).reshape(len(rows), -1)
            if team == 'BALL':
                if out[section]['ball'] is not None:
                    raise ValueError('duplicate ball frameset')
                out[section]['ball'] = data
            elif team in (home, guest):
                if (team, person) in out[section]['players']:
                    raise ValueError('duplicate player frameset')
                out[section]['players'][(team, person)] = data
            # Other framesets (e.g. officials) are ignored deliberately.
            current, rows = None, []
            elem.clear()
    if any(out[s]['ball'] is None for s in SECTIONS):
        raise ValueError('ball frameset missing for a half')
    return out


def labels_from_ball(possession, status):
    """Competition semantics: D where the tracking marks the ball dead, else the owning team."""
    out = np.full(len(status), 'X', dtype='<U1')
    out[status == 0] = 'D'
    alive = status == 1
    out[alive & (possession == 1)] = 'H'
    out[alive & (possession == 2)] = 'A'
    return out


def assemble(parsed, home, guest, pitch):
    """Full match: half 1, one ungraded separator frame (period 0), half 2. Coordinates -> [0, 1]."""
    keys = sorted({k for s in SECTIONS for k in parsed[s]['players']},
                  key=lambda k: (k[0] != home, k[1]))
    teams = np.array([0 if team == home else 1 for team, _ in keys])
    length, width = pitch
    blocks = []
    for half, section in enumerate(SECTIONS, start=1):
        ball = parsed[section]['ball']
        frames = ball[:, 0].astype(int)
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError('ball frames must be contiguous within a half')
        n = len(frames)
        players = np.full((n, len(keys), 2), np.nan)
        for slot, key in enumerate(keys):
            data = parsed[section]['players'].get(key)
            if data is None or not len(data):
                continue
            idx = data[:, 0].astype(int) - frames[0]
            keep = (idx >= 0) & (idx < n)
            players[idx[keep], slot] = data[keep, 1:3]
        players[..., 0] = (players[..., 0] + length / 2) / length
        players[..., 1] = (players[..., 1] + width / 2) / width
        xy = np.column_stack([(ball[:, 1] + length / 2) / length, (ball[:, 2] + width / 2) / width])
        blocks.append({'players': players, 'ball': xy, 'z': ball[:, 3],
                       'labels': labels_from_ball(ball[:, 4].astype(int), ball[:, 5].astype(int)),
                       'periods': np.full(n, half)})
    sep = {'players': np.full((1, len(keys), 2), np.nan), 'ball': np.full((1, 2), np.nan),
           'z': np.full(1, np.nan), 'labels': np.array(['X']), 'periods': np.zeros(1, int)}
    parts = [blocks[0], sep, blocks[1]]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} | {'teams': teams}


def load(match, raw=RAW, cache=True):
    info, positions = paths(match, raw)
    digest = hashlib.sha256(positions.read_bytes()).hexdigest()[:16]
    cached = raw / 'cache' / f'{match}-{digest}.npz'
    if cache and cached.exists():
        with np.load(cached) as data:
            return {k: data[k] for k in data.files}
    pitch, home, guest = parse_info(info)
    arrays = assemble(parse_positions(positions, home, guest), home, guest, pitch)
    arrays['pitch'] = np.array(pitch)
    if cache:
        cached.parent.mkdir(exist_ok=True)
        np.savez(cached, **arrays)
    return arrays
