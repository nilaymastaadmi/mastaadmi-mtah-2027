#!/usr/bin/env bash
# v10 (EXPERIMENT-18 arm A) on the competition match, twice, then compare digests. LOCAL MACHINE ONLY.
#   bash v10/run.sh <data-dir>
# Prints aggregates and digests only. Writes work/v10-evaluation/{inputs,labels}_runN.* and v10_<stamp>_runN.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:?data dir with tactical/ and broadcast/}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT=work/v10-evaluation
for run in 1 2; do
  OMP_NUM_THREADS=1 .venv/bin/python -m v10.inputs --data-dir "$DATA_DIR" --out "$OUT/inputs_${STAMP}_run$run.npz"
  .venv-video/bin/python -m v10.predict --inputs "$OUT/inputs_${STAMP}_run$run.npz" --out "$OUT/labels_${STAMP}_run$run.npy"
  .venv/bin/python -m v10.write --labels "$OUT/labels_${STAMP}_run$run.npy" --out "$OUT/v10_${STAMP}_run$run.csv"
done
a=$(shasum -a 256 "$OUT/v10_${STAMP}_run1.csv" | cut -d' ' -f1)
b=$(shasum -a 256 "$OUT/v10_${STAMP}_run2.csv" | cut -d' ' -f1)
echo "run1 $a"
echo "run2 $b"
[ "$a" = "$b" ] && echo "IDENTICAL" || { echo "DIFFERENT"; exit 1; }
