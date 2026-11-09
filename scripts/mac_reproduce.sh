#!/usr/bin/env bash
# Local-machine reproduction of the frozen v6 submission candidate (v4, v2 fallbacks). Run on Nilay's Mac only.
#
#   bash scripts/mac_reproduce.sh [DATA_DIR]      # DATA_DIR contains tactical/ (default: data)
#
# Never paste the generated CSV, arrays or frames into a hosted model. Printing the CSV's SHA-256
# here is fine: it identifies the file but reveals nothing about the match.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:-data}"
PY=.venv/bin/python
export OMP_NUM_THREADS=2

step() { printf '\n== %s\n' "$1"; }

step "1/6 Environment"
if [ ! -x "$PY" ]; then
  echo "No .venv. Create it with Python 3.13 (3.13.16 preferred):"
  echo "  MTAH_PYTHON=\$(command -v python3.13) bash scripts/bootstrap.sh"
  exit 1
fi
"$PY" - <<'EOF'
import platform, sys, numpy, pandas, sklearn
need = {'numpy': '2.5.4', 'pandas': '3.1.0', 'sklearn': '1.9.1'}
have = {'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__}
print('python', platform.python_version(), have)
bad = {k: v for k, v in have.items() if v != need[k]}
if sys.version_info[:2] != (3, 13) or bad:
    sys.exit(f'Runtime mismatch {bad or platform.python_version()}: run .venv/bin/python -m pip install -r requirements.txt')
EOF

step "2/6 Snapshot and synthetic tests"
"$PY" scripts/check_snapshot.py
"$PY" -m unittest -q test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6 v8.test_v8

step "3/6 Cross-machine check on public DFL data (optional)"
if ls external/idsse/DFL_04_03_positions_raw_observed_*.xml >/dev/null 2>&1; then
  "$PY" -m v2.stability_check --compare
  if ls external/skillcorner-lfs/*.jsonl >/dev/null 2>&1; then "$PY" -m v4.stability_check --compare; "$PY" -m v6.stability_check --compare; fi
  # v8: aligns a simulated second view on two DFL matches with all cores; expect tens of minutes. Resumable per match.
  if [ -f work/v8-evaluation/reference-digests.json ]; then OMP_NUM_THREADS=1 "$PY" -m v8.stability_check --compare; fi
else
  echo "Skipped: external/idsse not present (see HANDOFF.md to fetch the 14 public DFL files)."
fi

step "4/6 Data present"
for f in "$DATA_DIR/tactical/BAR-ATH_tactical_atd.txt" "$DATA_DIR/tactical/BAR-ATH_tactical_atd.xml"; do
  [ -f "$f" ] || { echo "Missing $f (pass the folder that contains tactical/)"; exit 1; }
done
echo "ok"

step "5/6 Generate twice and compare (v6 candidate; v4, v2 fallbacks)"
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p work/candidates
twice() {  # $1 module, $2 tag
  local r1="work/candidates/${2}_${stamp}_run1.csv" r2="work/candidates/${2}_${stamp}_run2.csv"
  "$PY" -m "$1" --data-dir "$DATA_DIR" --out "$r1"
  "$PY" -m "$1" --data-dir "$DATA_DIR" --out "$r2"
  local a b; a=$(shasum -a 256 "$r1" | cut -d' ' -f1); b=$(shasum -a 256 "$r2" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "$2 runs differ ($a vs $b): do not submit; report this."; exit 1; }
  echo "$2 identical: $a"
  eval "${2}_file=$r1; ${2}_sha=$a"
}
if [ -f "$DATA_DIR/broadcast/BAR-ATH_broadcast_atd.txt" ] && [ -f work/v8-evaluation/freeze.json ]; then
  OMP_NUM_THREADS=1 twice v8.infer v8   # broadcast-fused candidate (EXPERIMENT-13); alignment uses all cores
else
  echo "v8 skipped: broadcast assets or freeze.json missing"; v8_file=none; v8_sha=none
fi
twice v6.infer v6
twice v4.infer v4
twice v2.verified_inference v2

step "6/6 Perturbation checks on competition inputs (counts only)"
[ "$v8_file" = none ] || OMP_NUM_THREADS=1 "$PY" -m v8.stability_check --data-dir "$DATA_DIR"
"$PY" -m v6.stability_check --data-dir "$DATA_DIR"
"$PY" -m v4.stability_check --data-dir "$DATA_DIR"
"$PY" -m v2.stability_check --data-dir "$DATA_DIR"

printf '\nFused     (v8): %s\nSHA-256:        %s  (candidate only if the instrument checks and tag audit pass)\n' "$v8_file" "$v8_sha"
printf 'Candidate (v6): %s\nSHA-256:        %s\n' "$v6_file" "$v6_sha"
printf 'Fallback  (v4): %s\nSHA-256:        %s\n' "$v4_file" "$v4_sha"
printf 'Fallback  (v2): %s\nSHA-256:        %s\n' "$v2_file" "$v2_sha"
echo "Next: cp \"$v6_file\" Mastaadmi.csv && $PY starter-repo/evaluation/validate_submission.py Mastaadmi.csv"
