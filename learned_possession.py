"""Learn state probabilities on external games; restricted inference remains local."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import ROOT, features, decode, load_atd, write_submission
from external_validation import load_sample, proxy_labels, mask_ball, baseline, measure


def inputs(players, teams, ball, periods):
    costs, owner = features(players, teams, ball, periods)
    visible = np.isfinite(ball).all(axis=1)
    speed = np.full(len(ball), np.nan)
    valid = visible[1:] & visible[:-1] & (periods[1:] == periods[:-1])
    speed[1:][valid] = np.minimum(50., np.linalg.norm(np.diff(ball, axis=0)[valid] * [105., 68.], axis=1) * 25)
    moving = np.linalg.norm(np.diff(players, axis=0) * [105., 68.], axis=2) * 25
    moving = np.where(np.isfinite(moving), np.minimum(moving, 12.), np.nan)
    counts = np.sum(np.isfinite(moving), axis=1)
    mean_speed = np.r_[0., np.divide(np.nansum(moving, axis=1), counts,
                                    out=np.zeros(len(counts)), where=counts > 0)]
    mean_speed[np.r_[False, periods[1:] != periods[:-1]]] = 0.
    gap = np.zeros(len(ball))
    # Full gap duration uses future observations; this is an offline decoder.
    edges = np.r_[0, np.flatnonzero((visible[1:] != visible[:-1]) |
                                   (periods[1:] != periods[:-1])) + 1, len(ball)]
    for start, end in zip(edges[:-1], edges[1:]):
        if not visible[start]:
            gap[start:end] = min((end - start) / 25., 60.)
    outside = visible & ((ball < 0).any(axis=1) | (ball > 1).any(axis=1))
    x = np.column_stack([costs[:, :2] * 25., speed,
                         (costs[:, 2] * 25. == .25).astype(float),
                         gap, outside.astype(float), visible.astype(float), mean_speed])
    return x, owner


def condition_inputs(arrays, condition):
    players, teams, ball, periods = arrays[:4]
    observed = ball if condition == 'clean' else mask_ball(ball)
    return inputs(players, teams, observed, periods)


def predict(model, x, periods, penalty):
    probs = model.predict_proba(x)
    # Model classes are explicitly reordered into the decoder's H/A/D convention.
    probs = probs[:, [list(model.classes_).index(c) for c in ['H', 'A', 'D']]]
    return decode(-np.log(np.clip(probs, 1e-6, 1.)) / 25., periods, penalty)


def compare(model, arrays, gt, mask, penalties, stage):
    rows = []
    for condition in ['clean', 'masked']:
        x, owner = condition_inputs(arrays, condition)
        periods = arrays[3]
        outputs = {'baseline50': baseline(owner, periods)}
        outputs.update({f'learned{p:g}': predict(model, x, periods, p) for p in penalties})
        for name, pred in outputs.items():
            rows.append({'stage': stage, 'condition': condition, 'method': name,
                         'graded_frames': int(np.sum(gt[mask] != 'X')),
                         **measure(gt[mask], pred[mask], periods[mask])})
    return rows


def gate(rows, candidate):
    pairs = {c: {r['method']: r for r in rows if r['condition'] == c}
             for c in ['clean', 'masked']}
    gains = {c: x[candidate]['spellF1'] - x['baseline50']['spellF1'] for c, x in pairs.items()}
    macro = pairs['clean'][candidate]['macroF1'] - pairs['clean']['baseline50']['macroF1']
    return {'spellF1_deltas': gains, 'clean_macroF1_delta': macro,
            'provisional_gate_passed': all(v > 0 for v in gains.values()) and macro >= -.03}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['possession.py', 'external_validation.py', 'learned_possession.py']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'holdout', 'infer'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    folder = ROOT / 'work/learned-evaluation'
    folder.mkdir(exist_ok=True)
    if args.stage in ['dev', 'holdout']:
        marker = folder / (args.stage + '.started')
        with marker.open('x') as f:
            json.dump({'source_hashes': source_hashes()}, f)
    if args.stage == 'dev':
        arrays = load_sample(1)
        periods = arrays[3]
        gt = proxy_labels(arrays[4], periods)
        train = (periods == 1) & (gt != 'X')
        x = np.concatenate([condition_inputs(arrays, c)[0][train] for c in ['clean', 'masked']])
        y = np.tile(gt[train], 2)
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=50, learning_rate=.08, l2_regularization=1.,
                    early_stopping=False, random_state=20261028)
        model.fit(x, y)
        penalties = [0., .1, .5, 1.5]
        rows = compare(model, arrays, gt, periods == 2, penalties, 'development: game1 half2')
        means = {p: np.mean([r['spellF1'] for r in rows if r['method'] == f'learned{p:g}'])
                 for p in penalties}
        penalty = next(p for p in penalties if means[p] >= max(means.values()) - .005)
        report = {'rows': rows, 'selected_penalty': penalty,
                  'training_original_frames': int(train.sum()), 'source_hashes': source_hashes(),
                  **gate(rows, f'learned{penalty:g}')}
        with (folder / 'model.pkl').open('wb') as f:
            pickle.dump(model, f)
        report['model_sha256'] = hashlib.sha256((folder / 'model.pkl').read_bytes()).hexdigest()
        (folder / 'dev.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    else:
        dev = json.loads((folder / 'dev.json').read_text())
        if source_hashes() != dev['source_hashes']:
            raise SystemExit('Source changed since model freeze; require an explicit experiment amendment.')
        model_path = folder / 'model.pkl'
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != dev['model_sha256']:
            raise SystemExit('Model hash differs from development freeze')
        if not dev['provisional_gate_passed']:
            raise SystemExit('Development gate failed; holdout/inference are not permitted.')
        # Only load a locally trained, hash-verified pickle, never external pickles.
        with model_path.open('rb') as f:
            model = pickle.load(f)
        penalty = dev['selected_penalty']
        if args.stage == 'holdout':
            arrays = load_sample(2)
            gt = proxy_labels(arrays[4], arrays[3])
            rows = compare(model, arrays, gt, np.ones(len(gt), bool), [penalty], 'holdout game2')
            report = {'rows': rows, 'source_hashes': source_hashes(),
                      'selected_penalty': penalty, **gate(rows, f'learned{penalty:g}')}
            (folder / 'holdout.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        else:
            if not args.data_dir or not args.out:
                raise SystemExit('infer requires --data-dir and --out')
            holdout_path = folder / 'holdout.json'
            if not holdout_path.exists() or not json.loads(holdout_path.read_text())['provisional_gate_passed']:
                raise SystemExit('Held-out gate has not passed; inference is not permitted.')
            arrays = load_atd(args.data_dir)
            x, _ = inputs(*arrays)
            pred = predict(model, x, arrays[3], penalty)
            write_submission(str(args.out), pred)
            print('Wrote local candidate; no competition labels read.')
