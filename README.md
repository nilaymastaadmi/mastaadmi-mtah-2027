# More Than A Hack 2027, Phase 1: ball possession

This repository regenerates our possession file for the More Than A Hack 2027 Phase 1 challenge (Barcelona v Athletic, tactical video, 199,319 frames). The method and its validation are described in [`docs/technical-note.md`](docs/technical-note.md).

**Match data is not included and must not be committed.** Obtain the bundle from the organisers and place it as:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_atd.xml
```

## Regenerate the submission

Requires Python 3.13 (frozen with 3.13.16) and two virtual environments. `.venv` holds the packages pinned in `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1). `.venv-video` holds `requirements-v10.txt` (torch 2.15.0, numpy 2.5.4) for the neural network stage; the folder name is historical. On Linux, install the CPU build of torch with `--index-url https://download.pytorch.org/whl/cpu`.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .venv-video
.venv-video/bin/python -m pip install -r requirements-v10.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6 v8.test_v8 exp18.test_exp18 exp18.test_arm_b
bash v10/run.sh data
```

`v10/run.sh` runs three stages twice and compares the two CSVs: `v10.inputs` aligns the broadcast feed, fills missing ball positions and builds the model input; `v10.predict` runs the frozen network and decoder; `v10.write` writes the CSV and validates it with the organisers' validator. The prediction stage checks the pre-registered gate, the frozen model, its settings and every bound source file against the freeze before it runs. Nothing reads ground truth. The files appear in `work/v10-evaluation/` (`v10_<stamp>_run1.csv`); copy one to `Mastaadmi.csv`. The expected SHA-256 is in the technical note. The earlier v8 pipeline still runs with `OMP_NUM_THREADS=1 .venv/bin/python -m v8.infer --data-dir data --out v8.csv`.

The labels are identical on macOS arm64 and Linux x86 on 2.45 million public frames, but the network's probabilities are not bit-identical across machines (see the technical note, Section 6). The digest in the note is for macOS arm64.

## Verify the model from public data (optional)

The model was trained only on public matches from other competitions. To rebuild it and check that it reproduces the frozen predictions:

1. Download the 14 match-information and raw-positions XML files of the IDSSE/DFL dataset (figshare article 28196177 v1, CC BY 4.0). Do not download the event files.
2. Place them in `external/idsse/`.
3. Run:

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m v2.retrain
OMP_NUM_THREADS=2 .venv/bin/python -m v2.stability_check --compare
```

The full pre-registered experiment is in `work/EXPERIMENT-04.md` and `v2/experiment.py`, with recorded results in `work/v2-evaluation/` and `outputs/v2-experiment-04-results.md`. Each stage was one-shot and is guarded against reruns.

## Layout

| Path | Contents |
|---|---|
| `v10/` | Submitted pipeline: three stages (inputs, network and decoder, CSV writer), the frozen network `final.pt`, the run script and a cross-platform check |
| `exp18/` | Code v10 imports (geometry features, network, decoder) and the dropped possession-change detector, with tests |
| `v8/` | Broadcast tracking alignment and ball fusion (used by v10 for its input), the simulated-view experiments, stability checks, tests; previous submission |
| `v6/` | Per-frame model with ball-control features, used by v8; first fallback |
| `v4/` | Earlier method and fallback: team-shape and context features, frozen model, verified inference, stability checks |
| `v2/` | Earlier method and fallback: loader, degradation, comparators, experiment runner, verified inference, reproducibility checks, tests |
| `possession.py`, `learned_possession.py` | Shared feature extraction and decoder, from version 1; frozen |
| `hybrid_possession.py`, `verified_inference.py`, `external_validation.py`, `first_assessment.py` | Version 1 (fallback) and its earlier external evaluation; frozen |
| `work/` | Experiment registrations, frozen models, aggregate results |
| `starter-repo/` | Selected organiser code and docs, unmodified (MIT) |

## Attribution

- IDSSE/DFL data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0.
- SkillCorner open data: https://github.com/SkillCorner/opendata (MIT).
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data.
- Organiser starter code: https://github.com/metrica-sports/more-than-a-hack-2027 (MIT).
- See `THIRD_PARTY.md`.

Licence for our own code: MIT (see `LICENSE`). Vendored organiser files keep their own MIT notice, and datasets are not included.
