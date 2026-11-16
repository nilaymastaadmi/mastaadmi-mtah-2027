"""EXPERIMENT-22 candidate on the competition match (only after a registered PASS). LOCAL MACHINE ONLY, label-free.

    OMP_NUM_THREADS=1 .venv/bin/python -m exp22.candidate inputs --data-dir <data> --out <inputs.npz>
    .venv-video/bin/python -m exp22.candidate predict --inputs <inputs.npz> --out <labels.npy>
    .venv/bin/python -m v10.write --labels <labels.npy> --out <candidate.csv>
    OMP_NUM_THREADS=1 .venv/bin/python -m exp22.candidate adoption --data-dir <data> --v8 Mastaadmi.csv --cand <csv>

inputs: v8's tracking path once (tactical ATD, broadcast aligned and fused into missing ball frames), the 90 columns as
v10.inputs, and v8's frozen v6 evidence q_v8 on the float64 72 columns as v8.infer. Also checks that v8's decoder on
q_v8 reproduces the v8 submission labels frame for frame. predict: M final conv (hash-checked against
work/exp22-evaluation/freeze.json), blend 0.5 with q_v8, v8's decoder. adoption: exp19.adoption's definitions and fail
rules, written to work/exp22-evaluation/adoption.json. Never reads ground truth, tags or evaluation files (the tag
audit is v4.smart_tagging_audit, run separately). Prints aggregates only.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'work/exp22-evaluation'
MODEL = FOLDER / 'final.pt'      # frozen copy of external/exp22/M/models/final.pt (sha256 in freeze.json)
W_CONV = .5
V8_SETTING = (.5, 0.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BOUND = ['exp22/candidate.py', 'exp22/net.py', 'exp22/views.py', 'exp22/evaluate.py', 'exp18/arm_a.py',
         'exp18/targets.py', 'exp18/decode.py', 'exp18/geometry.py', 'v10/write.py', 'v6/infer.py', 'v6/control.py',
         'v3/features.py', 'v4/shape.py', 'v4/develop.py', 'v2/methods.py', 'v8/align.py', 'v8/infer.py',
         'v8/broadcast_audit.py', 'possession.py', 'starter-repo/evaluation/validate_submission.py',
         'work/EXPERIMENT-22.md', 'work/exp22/gate.json']


def bound_hashes():
    return {p: sha(ROOT / p) for p in BOUND}


def gate_passed():
    g = json.loads((ROOT / 'work/exp22/gate.json').read_text())
    if g['verdict'] != 'PASS':
        raise SystemExit(f"EXPERIMENT-22 gate verdict is {g['verdict']}; no candidate")


def read_csv_labels(path):
    return np.array([r.split(',')[1] for r in Path(path).read_text().splitlines()[1:]])


def inputs(args):
    gate_passed()
    from possession import load_atd
    from v2 import methods as mt
    from v3 import features as f3
    from v4.shape import team_shape
    from v6.control import control_features
    from v6.infer import columns, load_frozen
    from v8 import align as al
    from v8.broadcast_audit import load_broadcast
    from v8.infer import check_freeze
    from exp18 import decode as de
    from exp18.geometry import geometry
    check_freeze()
    players, teams, ball, periods = load_atd(args.data_dir)
    bp, bb = load_broadcast(args.data_dir)
    fused, source = al.fuse_ball(ball, bb, al.align(players, bp, periods))
    x72 = columns(f3.build(players, teams, fused, periods), team_shape(players, teams, periods)[0],
                  control_features(players, teams, fused, periods))
    q_v8 = mt.learned_probs(load_frozen(), x72)
    x = np.hstack([x72, geometry(players, teams, fused, periods)]).astype(np.float32)
    periods = np.asarray(periods)
    v8_labels = de.decode(q_v8, None, periods, *V8_SETTING)
    sub = read_csv_labels(ROOT / 'Mastaadmi.csv')
    diff = int(np.sum(sub != v8_labels)) if len(sub) == len(v8_labels) else -1
    if diff != 0:
        raise SystemExit(f'v8 reproduction failed: {diff} frames differ from Mastaadmi.csv')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=x, periods=periods, q_v8=q_v8)
    g = periods > 0
    print(json.dumps({'frames': int(len(periods)), 'graded': int(g.sum()), 'v8_frames_differing': diff,
                      'missing_ball_share': round(float((source[g] == -1).mean()), 3)}))


def predict(args):
    from exp22 import net
    from exp18 import decode as de
    freeze = json.loads((FOLDER / 'freeze.json').read_text())
    if freeze['model_sha256'] != sha(MODEL):
        raise SystemExit('M final model differs from freeze')
    if freeze['gate_verdict'] != 'PASS':
        raise SystemExit('freeze does not record a PASS')
    if freeze['source_hashes'] != bound_hashes():
        raise SystemExit('a bound source changed since freeze')
    d = np.load(args.inputs)
    model, scaler = net.load_model(MODEL)
    q_conv, c = net.predict(model, scaler, d['x'], d['periods'])
    q = W_CONV * q_conv.astype(float) + (1 - W_CONV) * d['q_v8'].astype(float)
    labels = de.decode(q, None, d['periods'], *V8_SETTING)
    np.save(args.out, labels)
    np.savez(args.out.with_suffix('.probs.npz'), q=q.astype(np.float32), c=c, q_conv=q_conv)
    g = d['periods'] > 0
    print(json.dumps({'graded': int(g.sum()), 'd_share': round(float(np.mean(labels[g] == 'D')), 4)}))


def adoption(args):
    from possession import load_atd
    from v8 import align as al
    from v8.broadcast_audit import load_broadcast
    from exp18.geometry import _velocity
    from exp19.adoption import quiet_mask, stats
    players, teams, ball, periods = load_atd(args.data_dir)
    bp, bb = load_broadcast(args.data_dir)
    fused, _ = al.fuse_ball(ball, bb, al.align(players, bp, periods))
    periods = np.asarray(periods)
    missing = ~np.isfinite(fused).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        speed = np.nanmean(np.linalg.norm(_velocity(players, periods), axis=2), axis=1)
    quiet = quiet_mask(missing, speed, periods)
    res = {'quiet_share_of_graded': float(quiet[periods > 0].mean())}
    for name, path in (('v8', args.v8), ('cand', args.cand)):
        lab = read_csv_labels(path)
        assert len(lab) == len(periods)
        res[name] = {**stats(lab, periods, quiet), 'csv_sha256': sha(path)}
    a, b = res['v8'], res['cand']
    extra, extra_q = b['starts'] - a['starts'], b['starts_in_quiet'] - a['starts_in_quiet']
    res['fail_amendment2'] = bool(b['possessions_per_minute'] >= 1.1 * a['possessions_per_minute']
                                  and extra > 0 and extra_q >= .5 * extra)
    res['fail_amendment3'] = bool(b['starts_in_quiet'] >= 1.5 * a['starts_in_quiet'])
    res['fail'] = res['fail_amendment2'] or res['fail_amendment3']
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / 'adoption.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('inputs', 'predict', 'adoption'))
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--v8', type=Path)
    p.add_argument('--cand', type=Path)
    a = p.parse_args()
    if a.out is not None and a.out.exists():
        raise SystemExit('Refusing to overwrite')
    {'inputs': inputs, 'predict': predict, 'adoption': adoption}[a.stage](a)
