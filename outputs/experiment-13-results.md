# EXPERIMENT-13 results (5 November 2026): broadcast-view ball fusion passes the analogue gate

Simulated second view of each DFL match (`v8/simulate.py`: offset 1234 frames, live/replay shots, players within 35 m of the ball with 10% dropout, ball visible in 70% of live frames, 0.15 m noise, x-mirrored). Tactical input = contested degradation; fused input = the same with missing ball filled from the aligned simulated view (`v8/align.py`). Fold models = EXPERIMENT-10 recipe retrained out of fold (cached in `external/models/`). Decoder and features exactly v6; four label conventions averaged.

| Match | Missing ball before | after | Alignment exact | Reference | Fused | Delta | macroF1 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.412 | 0.201 | 0.719 | 0.4842 | 0.5198 | +0.0356 | +0.0178 |
| J03WN1 | 0.411 | 0.200 | 0.675 | 0.4373 | 0.4896 | +0.0523 | +0.0290 |
| J03WOH | 0.414 | 0.228 | 0.741 | 0.5098 | 0.5382 | +0.0284 | +0.0051 |
| J03WOY | 0.420 | 0.211 | 0.762 | 0.4827 | 0.5308 | +0.0481 | +0.0217 |
| J03WPY | 0.414 | 0.224 | 0.715 | 0.5314 | 0.5780 | +0.0466 | +0.0070 |
| J03WQQ | 0.420 | 0.219 | 0.765 | 0.5072 | 0.5487 | +0.0415 | +0.0038 |
| J03WR9 | 0.412 | 0.215 | 0.695 | 0.5040 | 0.5530 | +0.0490 | +0.0066 |

Mean delta **+0.0431**, wins 7 / 7, mean macroF1 delta +0.0130; missing-ball share 0.415 → 0.214. Orientation recovered on every match (x-mirror). **All three gate checks passed** (`work/v8-analogue/selection.json`).

## What this does and does not show

- Recovering about half of the missing ball frames from a second view is worth about +0.04 spellF1 on the contested analogue, consistently across matches. This is the largest single gain since EXPERIMENT-06.
- The second view is simulated. Real broadcast tracking has detector and homography errors the simulation does not model; the real-data instrument checks in `work/EXPERIMENT-13.md` (Mac, label-free) decide whether the alignment is usable on the competition match. Those are outstanding.
- Disclosed deviation: the simulation noise is 0.15 m, not the 0.3 m written in the registration (set before any outcome while fixing the synthetic tests).

## Freeze

`work/v8-evaluation/freeze.json` binds the alignment sources (`v8/align.py`, `v8/infer.py`, `v8/broadcast_audit.py`, `work/EXPERIMENT-13.md`). `v8/infer.py` loads the frozen v6 model unchanged. Adoption requires, in order: the Mac instrument checks, `v8.stability_check --compare` (cloud reference digests) and `--data-dir` on the Mac (zero changes), two identical `v8.infer` runs, and tag agreement ≥ v6 − 0.02.

## Cloud stability check (5 Nov)

`v8.stability_check --compare` on the two DEVELOPMENT matches (J03WOH, J03WOY) with a simulated second view: under coordinate perturbations of 1e-12, 1e-10 and 1e-8 m, mapping changes and label changes were **zero** at every scale on both matches. The smallest residual-to-threshold margin was 6.7e-6 m (J03WOH) and 1.1e-5 m (J03WOY), far above any cross-platform rounding. Reference digests are in `work/v8-evaluation/reference-digests.json`; a matching `--compare` on the Mac shows it reproduces the Linux alignment bit-for-bit.
