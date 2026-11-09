"""Label-free broadcast alignment diagnostic. LOCAL MACHINE ONLY. Prints aggregates, never frames.

    OMP_NUM_THREADS=2 .venv/bin/python -m v8.broadcast_audit --data-dir data

Needs data/broadcast/BAR-ATH_broadcast_atd.{xml,txt}. Reads no labels. Reports: chosen orientation,
per-half offsets, aligned share of broadcast frames, ball agreement where both views see the ball,
and how many tactical missing-ball frames the broadcast could fill. Writes the alignment to
work/broadcast/alignment.npz (local only, ignored) for later fusion.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'starter-repo'))
from v8 import align as al  # noqa: E402


def load_broadcast(data_dir):
    from games import asset_paths, source_frames
    paths = asset_paths(data_dir=str(data_dir), source='broadcast')
    root = ET.parse(paths['atd_meta']).getroot()
    slots = len(list(root.iter('Player')))
    n = source_frames(source='broadcast')
    players = np.full((n, slots, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(paths['atd_raw']) as handle:
        for line in handle:
            fields = line.strip().split(':')
            if len(fields) != 3:
                raise ValueError('malformed broadcast ATD line')
            i = int(fields[0]) - 1
            if not 0 <= i < n:
                raise ValueError('broadcast frame out of bounds')
            groups = fields[1].split(';')
            if len(groups) != slots:
                raise ValueError('broadcast slot count differs from metadata')
            players[i] = [[float(v) for v in g.split(',')[:2]] for g in groups]
            ball[i] = [float(v) for v in fields[2].split(',')[:2]]
    return players, ball


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    from possession import load_atd
    tp, tt, tb, periods = load_atd(args.data_dir)
    bp, bb = load_broadcast(args.data_dir)
    res = al.align(tp, bp, periods)
    graded = periods > 0
    aligned = res['mapping'] >= 0
    median, count = al.agreement(tb, bb, res)
    fused, source = al.fuse_ball(tb, bb, res)
    missing = graded & ~np.isfinite(tb).all(axis=1)
    out = ROOT / 'work/broadcast'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'alignment.npz', mapping=res['mapping'], residual=res['residual'],
             flip=np.array(res['flip']), offsets=np.array([res['offsets'].get(1, -1), res['offsets'].get(2, -1)]))
    print(json.dumps({
        'orientation_flip_xy': list(res['flip']), 'offsets_by_half': res['offsets'],
        'broadcast_frames': int(len(bp)), 'aligned_share': round(float(aligned.mean()), 3),
        'aligned_residual_m_p50_p90': [round(float(np.percentile(res['residual'][aligned], q)), 2) for q in (50, 90)] if aligned.any() else None,
        'ball_agreement_median_m': None if median is None else round(median, 2), 'ball_agreement_frames': count,
        'tactical_missing_ball_graded': int(missing.sum()),
        'recovered_by_broadcast': int((source[graded] == 1).sum()),
        'missing_after_fusion_share': round(float((source[graded] == -1).mean()), 3),
        'missing_before_share': round(float(missing.mean()), 3)}, indent=1))
