#!/usr/bin/env bash
# v13 on the competition match, twice, then compare digests. LOCAL MACHINE ONLY, label-free.
#   bash v13/run.sh <data-dir>
# Stages: v13.inputs (.venv: v8 fusion path, warm-up fix, four views, 90 columns and v8 evidence per view), v13.predict
# (.venv-video: frozen network per view, four-view averages, 50/50 blend, v12 decoder, pre-kickoff fix), v10.write
# (.venv: CSV and the official validator). Each stage checks the freeze and the package pins first. Override the
# interpreters with PY and PYT. Prints digests only.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:?data dir with tactical/ and broadcast/}"
PY="${PY:-.venv/bin/python}"
PYT="${PYT:-.venv-video/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT=work/v13-evaluation/runs
mkdir -p "$OUT"
for run in 1 2; do
  OMP_NUM_THREADS=1 "$PY" -m v13.inputs --data-dir "$DATA_DIR" --out "$OUT/inputs_${STAMP}_run$run.npz"
  OMP_NUM_THREADS=3 "$PYT" -m v13.predict --inputs "$OUT/inputs_${STAMP}_run$run.npz" --out "$OUT/labels_${STAMP}_run$run.npy"
  "$PY" -m v10.write --labels "$OUT/labels_${STAMP}_run$run.npy" --out "$OUT/v13_${STAMP}_run$run.csv"
done
a=$(shasum -a 256 "$OUT/v13_${STAMP}_run1.csv" | cut -d' ' -f1)
b=$(shasum -a 256 "$OUT/v13_${STAMP}_run2.csv" | cut -d' ' -f1)
want=$("$PY" -c "import json; print(json.load(open('work/v13-evaluation/freeze.json'))['expected_csv_sha256'])")
echo "run1 $a"
echo "run2 $b"
[ "$a" = "$b" ] && echo "IDENTICAL" || { echo "DIFFERENT"; exit 1; }
[ "$a" = "$want" ] && echo "MATCHES FROZEN v13 ($want)" || { echo "DIFFERS FROM FROZEN v13 ($want)"; exit 1; }
echo "csv: $OUT/v13_${STAMP}_run1.csv"
