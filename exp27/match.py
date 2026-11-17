"""EXPERIMENT-27 on the competition match. LOCAL MACHINE ONLY, label-free (no truth, tags or evaluation files read
here; the tag audit is v4.smart_tagging_audit, run separately).

    OMP_NUM_THREADS=1 .venv/bin/python -B -m exp27.match inputs --data-dir <data> --out <inputs.npz> [--warmup]
    .venv-video/bin/python -B -m exp27.match predict --inputs <inputs.npz> --out <labels.npy> --arm T1|T2|v12
    .venv/bin/python -B -m exp27.match report --data-dir <data> --inputs <inputs.npz> --probs <probs.npz>

inputs: v12's tracking path (tactical ATD, broadcast aligned and fused into missing ball frames, v8.infer freeze
checked), the four views (exp27.views.transform), per view the 90 columns (float32) and the frozen v6 q (H/A mapped
back). Without --warmup it also checks that v8's decoder on V0's boosted q reproduces the frozen v8 label digest
(work/v12-evaluation/freeze.json). --warmup: second-half features, conv context and decoding start at the organiser
kickoff (frames 110,808 to 113,741 get period 0, then D from v12.kickoff). predict: v12's frozen network (sha256
checked), mean over views, blend, v12's decoder, pre-kickoff fix. report: view disagreement and possession counts.
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
from exp27.views import VIEWS, TEAM_SWAP, BACK, transform  # noqa: E402

MODEL = ROOT / 'external/exp22/M/models/final.pt'
MODEL_SHA = 'e68c5767a8b7fc080cdc71df7451f6abf8f5e15b730ff71944c318d086650acf'
START, KICKOFF = 110808, 113742


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracking(data_dir):
    from possession import load_atd
    from v8 import align as al
    from v8.broadcast_audit import load_broadcast
    from v8.infer import check_freeze
    check_freeze()
    players, teams, ball, periods = load_atd(data_dir)
    bp, bb = load_broadcast(data_dir)
    fused, _ = al.fuse_ball(ball, bb, al.align(players, bp, periods))
    return players, np.asarray(teams), fused, np.asarray(periods)


def warm(periods):
    from v12.kickoff import window
    assert window() == (START, KICKOFF)
    out = periods.copy()
    assert (periods[START:KICKOFF] == 2).all() and periods[KICKOFF] == 2 and periods[START - 1] != 2, 'half edges moved'
    out[START:KICKOFF] = 0
    return out


def inputs(a):
    from exp22 import views as vw
    from v2 import methods as mt
    from v6.infer import load_frozen
    from v12.inputs import v8_labels_sha
    players, teams, ball, periods = tracking(a.data_dir)
    pf = warm(periods) if a.warmup else periods
    model = load_frozen()
    out = {'periods': periods, 'periods_f': pf}
    for k in VIEWS:
        p, t, b = transform(players, teams, ball, k)
        x72, x, _ = vw.features(p, t, b, pf)
        q = mt.learned_probs(model, x72)
        out[f'x_{k}'], out[f'q_gbt_{k}'] = x.astype(np.float32), (q[:, BACK] if TEAM_SWAP[k] else q)
    if not a.warmup:
        want = json.loads((ROOT / 'work/v12-evaluation/freeze.json').read_text())['v8_labels_sha256']
        if v8_labels_sha(out['q_gbt_V0'], periods) != want:
            raise SystemExit('V0 boosted q does not reproduce the frozen v8 labels')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(json.dumps({'frames': int(len(periods)), 'graded_features': int((pf > 0).sum()), 'warmup': a.warmup,
                      'v8_base_reproduced': (not a.warmup) or 'not checked (warm-up changes the features)'}))


def predict(a):
    from exp22 import net
    from exp18 import decode as de
    from v12.kickoff import kickoff_fix
    if sha(MODEL) != MODEL_SHA:
        raise SystemExit('final model differs from v12 (e68c5767)')
    d = np.load(a.inputs)
    model, scaler = net.load_model(MODEL)
    pf = d['periods_f']
    conv = {}
    for k in VIEWS:
        q, _ = net.predict(model, scaler, d[f'x_{k}'], pf)
        conv[k] = (q[:, BACK] if TEAM_SWAP[k] else q).astype(float)
    gbt = {k: d[f'q_gbt_{k}'].astype(float) for k in VIEWS}
    mc, mg = sum(conv.values()) / 4, sum(gbt.values()) / 4
    qs = {'v12': .5 * conv['V0'] + (1 - .5) * gbt['V0'], 'T1': .5 * mc + .5 * gbt['V0'], 'T2': .5 * mc + .5 * mg}
    labels = {arm: kickoff_fix(de.decode(q, None, pf, .5, 0.)) for arm, q in qs.items()}
    np.save(a.out, labels[a.arm])
    np.savez(a.out.with_suffix('.probs.npz'), **{f'conv_{k}': conv[k].astype(np.float32) for k in VIEWS},
             **{f'labels_{arm}': lab for arm, lab in labels.items()})
    g = d['periods'] > 0
    print(json.dumps({'arm': a.arm, 'graded': int(g.sum()), 'd_share': round(float(np.mean(labels[a.arm][g] == 'D')), 4)}))


def median_possession_s(labels, periods):
    lab = np.where(periods > 0, labels, 'X')
    edges = np.r_[0, np.flatnonzero((lab[1:] != lab[:-1]) | (periods[1:] != periods[:-1])) + 1, len(lab)]
    runs = [b - a for a, b in zip(edges[:-1], edges[1:]) if lab[a] in ('H', 'A')]
    return float(np.median(runs)) / 25.


def report(a):
    from exp18.geometry import _velocity
    from exp19.adoption import quiet_mask, stats
    from exp22.candidate import read_csv_labels
    players, teams, ball, periods = tracking(a.data_dir)
    missing = ~np.isfinite(ball).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        speed = np.nanmean(np.linalg.norm(_velocity(players, periods), axis=2), axis=1)
    quiet = quiet_mask(missing, speed, periods)
    d, pr = np.load(a.inputs), np.load(a.probs)
    g = periods > 0
    conv = {k: pr[f'conv_{k}'] for k in VIEWS}
    gbt = {k: d[f'q_gbt_{k}'] for k in VIEWS}

    def disagree(qs):
        am = np.stack([qs[k].argmax(1)[g] for k in VIEWS])
        return round(float(np.mean((am != am[0]).any(axis=0))), 4)

    blend = {k: .5 * conv[k].astype(float) + .5 * gbt[k] for k in VIEWS}
    out = {'argmax_disagree_any_two_views': {'conv': disagree(conv), 'boosted': disagree(gbt), 'blend': disagree(blend)},
           'pairwise_vs_V0_blend': {k: round(float(np.mean(blend[k].argmax(1)[g] != blend['V0'].argmax(1)[g])), 4)
                                    for k in VIEWS[1:]}}
    sub = read_csv_labels(ROOT / 'Mastaadmi.csv')
    out['v12_pipeline_reproduces_Mastaadmi_csv'] = bool(len(sub) == len(pr['labels_v12'])
                                                        and np.array_equal(sub, pr['labels_v12']))
    v8 = read_csv_labels(ROOT / 'work/candidates/Mastaadmi_v8_bb0ba6f5.csv')
    arms = {'v8 (bb0ba6f5)': v8, **{arm: pr[f'labels_{arm}'] for arm in ('v12', 'T1', 'T2')}}
    out['arms'] = {arm: {**stats(lab, periods, quiet), 'median_possession_s': round(median_possession_s(lab, periods), 2)}
                   for arm, lab in arms.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('inputs', 'predict', 'report'))
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--probs', type=Path)
    p.add_argument('--arm', choices=('v12', 'T1', 'T2'), default='T1')
    p.add_argument('--warmup', action='store_true')
    a = p.parse_args()
    if a.stage != 'report' and a.out is not None and a.out.exists():
        raise SystemExit('Refusing to overwrite')
    {'inputs': inputs, 'predict': predict, 'report': report}[a.stage](a)
