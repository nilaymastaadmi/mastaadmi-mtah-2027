"""Write work/v10-evaluation/freeze.json once the EXPERIMENT-18 arm A gate has passed and the final model exists.

    .venv-video/bin/python -m v10.freeze      (imports torch via v10.predict)

The decoder setting is the best grid setting on the whole out-of-fold set by the registered selection score
(work/exp18/result-A.json `grid_full_oof`, ties to the earlier setting in grid order).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp18 import decode as de  # noqa: E402
from v10.predict import FOLDER, MODEL, bound_hashes, sha  # noqa: E402

if __name__ == '__main__':
    if (FOLDER / 'freeze.json').exists():
        raise SystemExit('freeze.json exists; refusing to overwrite')
    gate = json.loads((ROOT / 'work/exp18/gate-A.json').read_text())
    if not gate['passed']:
        raise SystemExit('arm A gate did not pass; nothing to freeze')
    grid = json.loads((ROOT / 'work/exp18/result-A.json').read_text())['grid_full_oof']
    scores = [grid[str(list(s))] for s in de.GRID]
    setting = list(de.GRID[scores.index(max(scores))])
    freeze = {'arm': 'A', 'setting': setting, 'model_sha256': sha(MODEL), 'source_hashes': bound_hashes()}
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / 'freeze.json').write_text(json.dumps(freeze, indent=1) + '\n')
    print(json.dumps(freeze, indent=1))
