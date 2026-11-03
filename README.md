# More Than A Hack 2027, Phase 1: ball possession

This repository regenerates our possession file for the More Than A Hack 2027 Phase 1 challenge (Barcelona v Athletic, tactical video, 199,319 frames). The method and its validation are described in [`docs/technical-note.md`](docs/technical-note.md).

**Match data is not included and must not be committed.** Obtain the bundle from the organisers and place it as:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
```

## Regenerate the submission

Requires Python 3.13. The model was frozen with numpy 2.5.4, pandas 3.1.0 and scikit-learn 1.9.1, which are pinned in `requirements.txt`.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4
OMP_NUM_THREADS=2 .venv/bin/python -m v4.infer --data-dir data --out submission.csv
```

`v4.infer` checks the frozen model, settings, every bound source file and the package versions against the pre-registered freeze. It then writes the CSV, validates it with the organisers' validator, and prints its SHA-256. It never reads ground truth.

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
| `v4/` | Submitted method: team-shape and context features, frozen model, verified inference, stability checks |
| `v2/` | Previous method and fallback: loader, degradation, comparators, experiment runner, verified inference, reproducibility checks, tests |
| `possession.py`, `learned_possession.py` | Shared feature extraction and decoder, from version 1; frozen |
| `hybrid_possession.py`, `verified_inference.py`, `external_validation.py`, `first_assessment.py` | Version 1 (fallback) and its earlier external evaluation; frozen |
| `work/` | Experiment registrations, frozen models, aggregate results |
| `starter-repo/` | Selected organiser code and docs, unmodified (MIT) |

## Attribution

- IDSSE/DFL data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0.
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data.
- Organiser starter code: https://github.com/metrica-sports/more-than-a-hack-2027 (MIT).
- See `THIRD_PARTY.md`.

Licence for our own code: MIT (see `LICENSE`). Vendored organiser files keep their own MIT notice, and datasets are not included.
