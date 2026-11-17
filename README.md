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

Requires Python 3.13 (frozen with 3.13.16). The submitted file is v14 (sha256 `ace6e2f701aed67ca0715d79f7ccc4f571ee599983e112a6d8d01aca4336cd30`). It needs two environments: `.venv` from `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1) and `.venv-video` from `requirements-v12.txt` (torch 2.15.0, numpy 2.5.4; on Linux use `--index-url https://download.pytorch.org/whl/cpu`).

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .venv-video
.venv-video/bin/python -m pip install -r requirements-v12.txt
bash v14/run.sh data
```

`v14/run.sh` checks the freeze (`work/v14-evaluation/freeze.json`: hashes of the network and the gradient-boosted model, bound source hashes, package versions), then runs the inputs (v8 fusion path, second half from the organiser kickoff, four mirrored copies with their features and v6 probabilities), the frozen network on each copy, the four-copy averages, the blend, the decoder, the second-half pre-kickoff step and the CSV writer twice, compares the two CSV digests and prints the SHA-256 and the path of the CSV (`work/v14-evaluation/runs/`). It never reads ground truth. Regeneration needs only the frozen weights (`v14/final.pt` and the v6 model) and the match data. v14 is v13 with the network retrained on 64 PFF World Cup matches added to its training data, so it has no score on a provider it has never seen; the technical note (Section 4.5) gives the leave-one-provider-out test that stands in for one. `bash v13/run.sh data` still regenerates the previous submission, v13 (`045ed8a5e189ece711f6996c7286024006f8a558440fc6e21919bc1dd008dac6`), and `bash v12/run.sh data` v12 (`0cdc7351e1bcfeb345f4a4a3549c0b2b5d1abe62a2aaa381c10b9a9a93879d7a`).

The exact SHA-256 has been produced on macOS arm64 (Apple M5, macOS, Python 3.13.16). A Linux check ran in our development repository on public external matches only, because the competition data may not go to a cloud runner. v14 result (GitHub Actions run 44076804695; Ubuntu x86_64, Python 3.13.16, torch 2.15.0+cpu, numpy 2.5.4) on the same 17 public views (2,454,886 frames): the labels match the Mac reference on every frame. The rebuilt inputs differ in the last float bits on all 17 views and the four-copy averaged probabilities by at most 5.2e-7, which changes no label (`work/v14-evaluation/portability-compare.json`). v13 result (GitHub Actions run 43966502242; Ubuntu x86_64, Python 3.13.16, torch 2.15.0+cpu, numpy 2.5.4) on the same 17 public views (2,454,886 frames): the labels match the Mac reference on every frame. The rebuilt inputs differ in the last float bits on all 17 views and the four-copy averaged probabilities by at most 8.8e-7, which changes no label (`work/v13-evaluation/portability-compare.json`). v12 result (GitHub Actions run 43799963735; Ubuntu x86_64, Python 3.13.16, torch 2.15.0+cpu, numpy 2.5.4) on 17 public views (7 DFL clean, 7 DFL with gaps, 3 SkillCorner; 2,454,886 frames): the labels match the Mac reference on every frame. The rebuilt inputs differ in the last float bits on all 17 views and the blended probabilities by at most 6.6e-7, which changes no label (`work/v12-evaluation/portability-compare.json`).

### v8 (base and fallback)

v8 is the base pipeline (tactical tracking, broadcast-ball fusion, v6 gradient-boosted model, Viterbi decoder) and the previous submission (sha256 `bb0ba6f55369f862a6ea2151882b82691a7331d6d98733bf6fe6225d08c4034e`). It still regenerates in `.venv`:

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6 v8.test_v8 exp18.test_exp18 exp18.test_arm_b exp22.test_exp22 v12.test_kickoff
OMP_NUM_THREADS=1 .venv/bin/python -m v8.infer --data-dir data --out v8.csv
```

`v8.infer` checks the pre-registered gate, the frozen v6 model, its settings, every bound source file and the package versions against the freeze. It aligns the broadcast feed, fills missing ball positions, decodes, writes the CSV, validates it with the organisers' validator, and prints its SHA-256. It never reads ground truth. On the public external matches the v8 pipeline gives identical labels on macOS arm64 and Linux x86.

### v10 (tested, withdrawn)

v10, a temporal network with about 4.4 s of context, was withdrawn after it lost to v8 on 64 PFF matches (technical note, Section 4). It still runs with `.venv-video` and `bash v10/run.sh data`.

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
| `v14/` | Submitted pipeline: v13's steps with the network retrained with PFF (`final.pt`), freeze checks, portability check, run script |
| `exp29/` | v14's training run (`train.py`: recipe of `exp22/net.py` with the PFF matches added); its scoring scripts need private caches and are not included |
| `v13/` | Previous submission: inputs (v8 fusion path, second half from the organiser kickoff, four mirrored copies), frozen network and v6 model on each copy, four-copy averages, blend, decoder, pre-kickoff step and CSV writer, freeze checks, portability check, run script |
| `exp27/` | v13's and v14's mirrored copies (`views.py`) and match path (`match.py`) |
| `v12/` | Earlier submission: inputs (v8 fusion path and v6 probabilities), frozen network `final.pt` (also used by v13), blend, decoder, second-half pre-kickoff step and CSV writer, freeze and version checks, portability check, run script |
| `exp22/` | v12's network (`net.py`), training views (`views.py`), Metrica labels (`metrica.py`) and tests; its scoring scripts need private caches and are not included |
| `v8/` | Base pipeline and fallback: broadcast tracking alignment and ball fusion, verified inference, the simulated-view experiments, stability checks, tests |
| `v10/` | Tested, withdrawn: three stages (inputs, network and decoder, CSV writer), the frozen network `final.pt`, the run script and a cross-platform check |
| `exp18/` | Code v10 imports (geometry features, network, decoder) and the dropped possession-change detector, with tests |
| `v6/` | Per-frame model with ball-control features, used by v8 |
| `v4/` | Earlier method and fallback: team-shape and context features, frozen model, verified inference, stability checks |
| `v2/` | Earlier method and fallback: loader, degradation, comparators, experiment runner, verified inference, reproducibility checks, tests |
| `possession.py`, `learned_possession.py` | Shared feature extraction and decoder, from version 1; frozen |
| `hybrid_possession.py`, `verified_inference.py`, `external_validation.py`, `first_assessment.py` | Version 1 (fallback) and its earlier external evaluation; frozen |
| `work/` | Experiment registrations, frozen models, aggregate results |
| `starter-repo/` | Selected organiser code and docs, unmodified (MIT) |

## Attribution

- IDSSE/DFL data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0.
- SkillCorner open data: https://github.com/SkillCorner/opendata (MIT).
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data. Used for training the v12 network; labels come from our own event rule.
- PFF FC World Cup 2022, released by PFF FC on request (https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data): used for testing (v8 to v13) and to train v14's network; not included and not redistributed; only v14's trained weights are published.
- Organiser starter code: https://github.com/metrica-sports/more-than-a-hack-2027 (MIT).
- See `THIRD_PARTY.md`.

Licence for our own code: MIT (see `LICENSE`). Vendored organiser files keep their own MIT notice, and datasets are not included.
