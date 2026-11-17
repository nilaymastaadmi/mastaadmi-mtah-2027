#!/usr/bin/env bash
# v14 on the competition match, twice, then compare digests. LOCAL MACHINE ONLY, label-free.
#   bash v14/run.sh <data-dir>
# Stages: v14.inputs (.venv: v8 fusion path, warm-up fix, four views, 90 columns and v8 evidence per view), v14.predict
# (.venv-video: network v14/final.pt per view, four-view averages, 50/50 blend, v12 decoder, pre-kickoff fix), v10.write
# (.venv: CSV and the official validator). Each stage checks the freeze and the package pins first. Override the
# interpreters with PY and PYT. Prints digests only.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:?data dir with tactical/ and broadcast/}"
PY="${PY:-.venv/bin/python}"
PYT="${PYT:-.venv-video/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT=work/v14-evaluation/runs
mkdir -p "$OUT"
for run in 1 2; do
  OMP_NUM_THREADS=1 "$PY" -m v14.inputs --data-dir "$DATA_DIR" --out "$OUT/inputs_${STAMP}_run$run.npz"
  OMP_NUM_THREADS=3 "$PYT" -m v14.predict --inputs "$OUT/inputs_${STAMP}_run$run.npz" --out "$OUT/labels_${STAMP}_run$run.npy"
  "$PY" -m v10.write --labels "$OUT/labels_${STAMP}_run$run.npy" --out "$OUT/v14_${STAMP}_run$run.csv"
done
a=$(shasum -a 256 "$OUT/v14_${STAMP}_run1.csv" | cut -d' ' -f1)
b=$(shasum -a 256 "$OUT/v14_${STAMP}_run2.csv" | cut -d' ' -f1)
want=$("$PY" -c "import json; print(json.load(open('work/v14-evaluation/freeze.json'))['expected_csv_sha256'])")
echo "run1 $a"
echo "run2 $b"
[ "$a" = "$b" ] && echo "IDENTICAL" || { echo "DIFFERENT"; exit 1; }
[ "$a" = "$want" ] && echo "MATCHES FROZEN v14 ($want)" || { echo "DIFFERS FROM FROZEN v14 ($want)"; exit 1; }
echo "csv: $OUT/v14_${STAMP}_run1.csv"
