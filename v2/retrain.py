"""Rebuild the frozen v2 model from public DFL development data and verify it reproduces predictions.

Uses exactly the registered training function (v2.experiment.train) on J03WOH, J03WOY and J03WPY.
Never overwrites work/v2-evaluation/model.pkl and runs no scoring (no labels of held-out matches are
read). Passing means the retrained model yields identical v2 labels to the frozen model on every
public reference view in work/v2-evaluation/reference-digests.json.

    OMP_NUM_THREADS=2 .venv/bin/python -m v2.retrain     # needs external/idsse (see HANDOFF.md)
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import degrade as dg, experiment as ex, idsse, methods as mt  # noqa: E402

if __name__ == '__main__':
    dev = json.loads((ROOT / 'work/v2-evaluation/dev.json').read_text())
    weight, penalty = dev['selected']['blend_learned']
    stored = json.loads((ROOT / 'work/v2-evaluation/reference-digests.json').read_text())['digests']
    loaded = {m: idsse.load(m) for m in idsse.DEVELOPMENT}
    model = ex.train(list(idsse.DEVELOPMENT), loaded)
    mismatched = []
    for key, digest in stored.items():
        match, condition = key.split('/')
        a = loaded.get(match) or idsse.load(match)
        players, ball = dg.degrade(a, match, condition, 'eval')
        obs = mt.observe(players, a['teams'], ball, a['periods'])
        q = mt.learned_probs(model, mt.learned_inputs(players, a['teams'], ball, a['periods']))
        labels = mt.blend(q, mt.prior50(obs, a['periods']), a['periods'], weight, penalty)
        if hashlib.sha256(''.join(labels).encode()).hexdigest() != digest:
            mismatched.append(key)
    print(f'retrained model reproduces {len(stored) - len(mismatched)}/{len(stored)} reference views')
    if mismatched:
        raise SystemExit(f'mismatched views: {mismatched}')
