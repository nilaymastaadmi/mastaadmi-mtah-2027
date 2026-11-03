"""Blend fixed baseline evidence with the frozen external-only learned model."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from possession import ROOT, baseline, decode, load_atd, write_submission
from external_validation import load_sample, proxy_labels, measure
from learned_possession import condition_inputs, inputs, gate


def predict(model, x, owner, periods, weight, penalty):
    reference = baseline(owner, periods)
    probs = model.predict_proba(x)
    probs = probs[:, [list(model.classes_).index(c) for c in 'HAD']]
    prior = np.full_like(probs, .01)
    for k, label in enumerate('HAD'):
        prior[reference == label, k] = .98
    mixture = weight * probs + (1 - weight) * prior
    return decode(-np.log(np.clip(mixture, 1e-6, 1.)) / 25., periods, penalty)


def hashes():
    files = ['possession.py', 'external_validation.py', 'learned_possession.py',
             'hybrid_possession.py', 'starter-repo/starter/baseline.py',
             'starter-repo/evaluation/possession_metrics.py', 'starter-repo/games.py']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'holdout', 'infer'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    folder = ROOT / 'work/hybrid-evaluation'
    folder.mkdir(exist_ok=True)
    original = json.loads((ROOT / 'work/learned-evaluation/dev.json').read_text())
    model_path = ROOT / 'work/learned-evaluation/model.pkl'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != original['model_sha256']:
        raise SystemExit('Frozen model hash mismatch')
    with model_path.open('rb') as handle:
        model = pickle.load(handle)
    if args.stage == 'dev':
        settings = [(w, p) for w in [.1, .25, .5] for p in [0., .1, .5]]
    else:
        dev = json.loads((folder / 'dev.json').read_text())
        if dev['selected'] is None or dev['source_hashes'] != hashes():
            raise SystemExit('No passing development candidate, or source changed since freeze')
        settings = [tuple(dev['selected'])]
    if args.stage == 'infer':
        held = json.loads((folder / 'holdout.json').read_text())
        if not held['gate']['provisional_gate_passed'] or held['source_hashes'] != hashes():
            raise SystemExit('Holdout gate or frozen-source check failed')
        if not args.out or not args.data_dir:
            raise SystemExit('infer requires --out and --data-dir')
        arrays = load_atd(args.data_dir)
        x, owner = inputs(*arrays)
        pred = predict(model, x, owner, arrays[3], *settings[0])
        write_submission(str(args.out), pred)
        print('Wrote local hybrid candidate without reading competition labels.')
    else:
        with (folder / (args.stage + '.started')).open('x') as f:
            json.dump({'source_hashes': hashes(), 'model_sha256': original['model_sha256']}, f)
        arrays = load_sample(1 if args.stage == 'dev' else 2)
        gt = proxy_labels(arrays[4], arrays[3])
        subset = arrays[3] == 2 if args.stage == 'dev' else np.ones(len(gt), bool)
        rows = []
        for condition in ['clean', 'masked']:
            x, owner = condition_inputs(arrays, condition)
            reference = baseline(owner, arrays[3])
            predictions = {'baseline50': reference}
            predictions.update({f'hybrid{w:g}_{p:g}': predict(model, x, owner, arrays[3], w, p)
                                for w, p in settings})
            for name, pred in predictions.items():
                rows.append({'condition': condition, 'method': name,
                             'graded_frames': int(np.sum(gt[subset] != 'X')),
                             **measure(gt[subset], pred[subset], arrays[3][subset])})
        eligible = []
        for w, p in settings:
            name = f'hybrid{w:g}_{p:g}'
            if gate(rows, name)['provisional_gate_passed']:
                eligible.append((w, p, np.mean([r['spellF1'] for r in rows if r['method'] == name])))
        selected = None
        if eligible:
            best = max(x[2] for x in eligible)
            selected = next([w, p] for w, p, score in eligible if score >= best - .005)
        report = {'stage': args.stage, 'rows': rows, 'selected': selected,
                  'source_hashes': hashes(), 'model_sha256': original['model_sha256']}
        if args.stage == 'holdout':
            w, p = settings[0]
            report['gate'] = gate(rows, f'hybrid{w:g}_{p:g}')
        (folder / (args.stage + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
