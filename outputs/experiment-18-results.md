# EXPERIMENT-18 results: finding the short possessions

Registration: `work/EXPERIMENT-18.md`, sha256 636e1916…d0ef, written 7 November before any outcome. Code: `exp18/`
(shared geometry, views, decoder, evaluator, arm A, arm B) and `v10/`. All numbers are out of fold on external
matches (5 match-grouped folds, `v4.develop.fold_of`), scored with the official scorer under the 4 conventions. The
reference R is v8's own stored out-of-fold probabilities with the v8 decoder, and it reproduces the bench numbers
exactly (largest difference 0.0).

## Verdict

- **Arm A (temporal conv net): PASS, 7 of 7 checks.** SkillCorner real +0.026 mean-of-4 spellF1 (0.512 to 0.538),
  wins on 20 of 20 matches. DFL gaps +0.106 (0.494 to 0.600), 7 of 7. DFL clean +0.089 (0.627 to 0.716), 7 of 7.
  controlled3 +0.027 (real) and +0.073 (gaps). Raw macroF1 on real tracking +0.012.
- **Arm B (change detector feeding the decoder): FAIL.** SkillCorner real +0.002 (13 of 20 wins), DFL gaps +0.005,
  clean +0.009. The detector ranks change frames well (AUC 0.87 real, 0.94 clean, 0.87 gaps) but its evidence moves
  almost nothing through the decoder.
- **Blend AB (arm A's q with arm B's change probability): passes the gate but is not preferred.** It gains +0.023 on
  real tracking, below arm A's +0.026, so the registered rule keeps the single arm A.

## What the numbers say, plainly

1. **On real tracking the gain is cleaner segmentation, not many more short possessions.** SkillCorner recall of
   true possessions under 3 s moves from 11% to 12% (raw) and from 16% to 19% (controlled3). Predicted possessions
   per match fall slightly (215 to 211), matched ones rise (162 to 165) and spurious ones fall (53.5 to 45.8), mostly
   fewer predictions over true dead time (9.2 to 5.0) and over the other team (26.2 to 23.8). The bench's +0.10 to
   +0.15 bound is far from reached on real tracking.
2. **On DFL the model does find short possessions.** Under-3 s recall goes from 17% to 31% on clean DFL and from 12%
   to 22% on gaps (raw); predicted possessions per match rise from 234 to 287 (clean) and from 235 to 272 (gaps),
   against 358 true. This is where the large gains come from.
3. **The change head was not used.** Every fold chose beta 0 (no change-aware switching) with P 1.0, the highest
   penalty in the grid. The gain comes from the sequence model's frame probabilities, decoded with a stiffer
   switch penalty. The P 1.0 choice sits at the edge of the registered grid; nothing beyond it was tried.
4. **How much of the real-tracking gain is the penalty?** A diagnostic outside the gate (`work/exp18/diag-penalty.json`):
   v8's own probabilities at P 1.0 score 0.516 on SkillCorner real (+0.004 over v8's P 0.5); arm A at v8's P 0.5
   scores 0.526 (+0.014). So the learned model carries most of the gain, but at v8's penalty its real-tracking margin
   (+0.014) would be under the +0.02 line. On DFL the gain does not depend on the penalty (gaps 0.600 at P 0.5).
5. **Real tracking gains far less than DFL** (+0.026 against +0.09 to +0.11). The competition match is real tracking,
   so the SkillCorner number is the relevant size. The DFL gaps training views come from the same gap simulation as
   the DFL gaps test views, so part of the gaps gain may be the model learning the simulation. One SkillCorner match
   is a near tie (+0.0002); the smallest clear win is +0.0075.
6. Arm B's failure fits the bench's finding that decoder changes alone barely move spellF1 (EXPERIMENT-08, +0.005):
   cheap switches only help where the frame evidence already favours the other team, and v8's evidence rarely does
   inside a short possession.

## Disclosed deviations and fixes

- Fable (read-only, 7 Nov) reviewed the exp18 code and found 3 process defects, none in a computation: arm A's fold
  lists would shift on a partial view cache (now refused), the grid score cache was not keyed by its inputs (now
  sha256-keyed, so arm B was re-scored; identical result), and the futility rule was not coded (run by hand:
  fold-0 SkillCorner raw macroF1 arm A 0.872 against R 0.859, no stop).
- One crash fix in arm A (the allowed defect fix): the final-model seed `[18, -1, epoch]` is rejected by numpy; it is
  now `[18, 5, epoch]`. Fold models 0 to 4 are unaffected (`work/exp18/defects.log`).
- The session stopped on 7 November at 15:07 (usage limit) with folds 0 and 1 done; fold 2 restarted from epoch 0 on
  9 November. Seeds are per fold and epoch, so the restart changes nothing.
- Not in the registration text but in the code from the start: a final LayerNorm before the two heads, and the
  scaler's median, IQR and missing-indicator columns are fitted on every 10th training frame.
- Folds 0 and 1 trained while other jobs shared the CPU (same 6-thread setting); only speed differs.

## Not checked

- Whether the real-tracking gain transfers to the competition's corrected tracking and label convention. The tag
  audit on the competition match is the only label-free check available (below).
- Raw-label training: the model is trained on provider labels that count deflections, like v8.

## v10 on the competition match (label-free, Mac only)

Arm A passed, so `v10/` was built as registered: the final model trained on all 27 external matches with the same
recipe (sha256 5fe16674…1bee), the decoder setting chosen on the whole out-of-fold set by the registered selection
score (P 1.0, beta 0), and v8's competition input path (tactical tracking with the broadcast ball fused in).

| Check | Result |
|---|---|
| Two full runs (`bash v10/run.sh <data-dir>`) | identical, sha256 `51bc7d93f1c32dbd6a65015a9a1451f9abedb66dce3d67f807036dc3b06eadcb` |
| Official validator | passed (199,319 rows) |
| Smart Tagging phase agreement (registered line 0.763 = v8 0.783 - 0.02) | 0.772, passes; v6 0.777 and v8 0.783 on the same audit |
| Input path | 4,936 ball frames recovered from the broadcast view, 37.6% still missing: the same as v8's audit |
| Against the current submission (v8, bb0ba6f5) | differs on 8.1% of graded frames; about 239 possessions against 255; D share 0.484 against 0.475 |

The CSV is `work/v10-evaluation/v10_20261110-004219_run1.csv` (run 2 is the identical copy). It was not copied to
`Mastaadmi.csv`; adoption is the parent session's call. Fable reviewed the v10 code read-only (no defects; three
minor notes applied before the freeze: bind the gate and result files, refuse to overwrite the freeze, assert the
input width).

What to expect if adopted: the out-of-fold real-tracking gain is +0.026 (about +0.014 of it without the stiffer
penalty), so a realistic private-score change is a few hundredths, not the DFL-sized +0.1. The tag agreement is
0.011 below v8's, inside the registered tolerance.

## Codex review (10 Nov): ADOPT AFTER FIXES, and what was done

1. **Freeze binding.** `v10/predict.py` now binds every repo file on the inference path (loaders, v3/v4/v6 feature
   code, v8 alignment, the CSV writer and validator, `requirements-v10.txt`), 35 files in all. A reordered feature
   column now fails the freeze.
2. **Grid cache key.** The key now covers q, c, the truth labels, the periods and the decoder and scorer code.
3. **Gate cardinality.** The gate refuses unless each setting holds exactly its registered matches (20 real, 7 clean,
   7 gaps), once each. After 2 and 3 every score was recomputed: `gate-A.json`, `gate-B.json`, `gate-AB.json` and all
   four `result-*.json` files are byte-identical to the first run.
4. **Model in a tracked path.** `v10/final.pt` (sha256 5fe16674…1bee, byte-identical to the trained file) loads with
   `torch.load(weights_only=True)`. `requirements-v10.txt` pins torch 2.15.0 and numpy 2.5.4 for the torch stage
   (`.venv-video` has no pandas; stages 1 and 3 use `requirements.txt`). Re-frozen; two new runs give the same CSV,
   sha256 51bc7d93…eadcb.
5. **Cross-platform check.** `v10/portability.py` runs the frozen model and decoder on 17 public views (7 DFL
   natural, 7 DFL contested, 3 SkillCorner; 2,454,886 frames) and writes per-view digests of the inputs, of q rounded
   to 1e-6 and of the decoded labels. Mac side: `work/v10-evaluation/portability-mac.json` (torch 2.15.0, Accelerate
   BLAS). Linux side: `.github/workflows/v10-portability.yml`, run 42689477932 on commit 60b6006 (ubuntu x86_64,
   torch 2.15.0+cpu with MKL and oneDNN, Python 3.13.16, numpy 2.5.4), built from the raw public downloads.
   Result (`work/v10-evaluation/portability-compare.json`): **decoded labels identical on 17 of 17 views, 0 of
   2,454,886 frames differ.** The pipeline is not bit-identical below the labels: the input feature digests differ
   on all 17 views, q rounded to 1e-6 differs on all 17, and the largest q difference is 9.5e-7. So the CSV
   reproduced here, but a near-tie in the decoder could still flip a frame on another machine; none did on 2.45
   million public frames. No rounding amendment was made.
6. **Nested selection (Codex P1, not fixed; disclosed here).** Fold-wise selection keeps fold f's own labels out of
   the choice of fold f's decoder, but the scores it uses come from the other folds' models, and those models were
   trained on fold f. So fold f's labels reach its decoder choice indirectly. Strict nested selection would refit
   every model without f; that was not done. All five folds chose the same setting (P 1.0, beta 0), so any inflation
   is limited to that one choice, and its size is not measured. The fair comparison takes v8's own best penalty
   instead of its registered one:

   | v8 decoded at | SkillCorner real: v8 / arm A / change / A wins | DFL gaps: change / wins | DFL clean: change / wins |
   |---|---|---|---|
   | P 0.25 | 0.502 / 0.538 / +0.036 / 20 of 20 | +0.100 / 7 of 7 | +0.078 / 7 of 7 |
   | P 0.5 (v8, the gate) | 0.512 / 0.538 / +0.026 / 20 of 20 | +0.106 / 7 of 7 | +0.089 / 7 of 7 |
   | P 1.0 (v8's best on real tracking) | 0.516 / 0.538 / +0.022 / 17 of 20 | +0.116 / 7 of 7 | +0.113 / 7 of 7 |

   Against v8 at P 1.0, arm A gains +0.022 on real tracking with 17 of 20 wins (worst match -0.001). By the
   registered selection score (mean of real and gaps), v8's best penalty is its own 0.5 (0.5031 against 0.5002 at
   P 1.0 and 0.5009 at P 0.25), so the gate's comparison stands. Source: `work/exp18/diag-fair-R.json`.

# Tables (generated by `python -m exp18.report`)

## Headline spellF1 (mean over matches)

| Setting | Arm | raw | c1 | c2 | c3 | mean of 4 | macroF1 raw |
|---|---|---:|---:|---:|---:|---:|---:|
| SkillCorner real | R (v8) | 0.460 | 0.499 | 0.537 | 0.554 | 0.512 | 0.855 |
| SkillCorner real | A (TCN) | 0.486 | 0.524 | 0.561 | 0.581 | 0.538 | 0.866 |
| SkillCorner real | B (change detector) | 0.463 | 0.501 | 0.539 | 0.554 | 0.514 | 0.855 |
| SkillCorner real | AB (A q + B c) | 0.484 | 0.522 | 0.558 | 0.576 | 0.535 | 0.866 |
| DFL clean | R (v8) | 0.544 | 0.595 | 0.664 | 0.707 | 0.627 | 0.916 |
| DFL clean | A (TCN) | 0.665 | 0.717 | 0.750 | 0.734 | 0.716 | 0.938 |
| DFL clean | B (change detector) | 0.562 | 0.612 | 0.669 | 0.703 | 0.636 | 0.918 |
| DFL clean | AB (A q + B c) | 0.674 | 0.725 | 0.752 | 0.730 | 0.720 | 0.939 |
| DFL gaps | R (v8) | 0.426 | 0.465 | 0.520 | 0.564 | 0.494 | 0.851 |
| DFL gaps | A (TCN) | 0.545 | 0.591 | 0.626 | 0.637 | 0.600 | 0.884 |
| DFL gaps | B (change detector) | 0.434 | 0.472 | 0.526 | 0.564 | 0.499 | 0.852 |
| DFL gaps | AB (A q + B c) | 0.550 | 0.593 | 0.620 | 0.628 | 0.598 | 0.884 |

## Gate

**A (TCN)**: PASS. Decoder chosen per fold (P, beta): 0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0), 4: (1.0, 0.0)

| Check | Value | Pass |
|---|---:|---|
| 1_real_mean4_ge_.02 | +0.0256 | yes |
| 2_gaps_mean4_ge_.02 | +0.1062 | yes |
| 3_c3_real_ge_0 | +0.0271 | yes |
| 3_c3_gaps_ge_0 | +0.0728 | yes |
| 4_real_wins_ge_15of20 | 20 of 20 | yes |
| 5_gaps_wins_ge_5of7 | 7 of 7 | yes |
| 6_clean_mean4_ge_-.02 | +0.0890 | yes |
| 7_real_macro_ge_-.02 | +0.0116 | yes |

DFL clean wins: 7 of 7.

**B (change detector)**: FAIL. Decoder chosen per fold (P, beta): 0: (1.0, 3.0), 1: (1.0, 3.0), 2: (1.0, 3.0), 3: (1.0, 3.0), 4: (1.0, 3.0)

| Check | Value | Pass |
|---|---:|---|
| 1_real_mean4_ge_.02 | +0.0021 | no |
| 2_gaps_mean4_ge_.02 | +0.0055 | no |
| 3_c3_real_ge_0 | +0.0007 | yes |
| 3_c3_gaps_ge_0 | +0.0003 | yes |
| 4_real_wins_ge_15of20 | 13 of 20 | no |
| 5_gaps_wins_ge_5of7 | 5 of 7 | yes |
| 6_clean_mean4_ge_-.02 | +0.0090 | yes |
| 7_real_macro_ge_-.02 | +0.0000 | yes |

DFL clean wins: 5 of 7.

**AB (A q + B c)**: PASS. Decoder chosen per fold (P, beta): 0: (1.0, 1.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (1.0, 0.0), 4: (1.0, 1.0)

| Check | Value | Pass |
|---|---:|---|
| 1_real_mean4_ge_.02 | +0.0228 | yes |
| 2_gaps_mean4_ge_.02 | +0.1040 | yes |
| 3_c3_real_ge_0 | +0.0226 | yes |
| 3_c3_gaps_ge_0 | +0.0639 | yes |
| 4_real_wins_ge_15of20 | 20 of 20 | yes |
| 5_gaps_wins_ge_5of7 | 7 of 7 | yes |
| 6_clean_mean4_ge_-.02 | +0.0929 | yes |
| 7_real_macro_ge_-.02 | +0.0111 | yes |

DFL clean wins: 7 of 7.

## Possession counts per match (mean over matches and the 4 conventions)

| Setting | Arm | True | Predicted | Matched | Spurious predicted |
|---|---|---:|---:|---:|---:|
| SkillCorner real | R (v8) | 290.5 | 215.3 | 161.8 | 53.5 |
| SkillCorner real | A (TCN) | 290.5 | 210.7 | 164.9 | 45.8 |
| SkillCorner real | B (change detector) | 290.5 | 216.8 | 162.8 | 54.0 |
| SkillCorner real | AB (A q + B c) | 290.5 | 216.1 | 166.4 | 49.6 |
| DFL clean | R (v8) | 357.8 | 234.0 | 213.9 | 20.1 |
| DFL clean | A (TCN) | 357.8 | 287.1 | 256.4 | 30.8 |
| DFL clean | B (change detector) | 357.8 | 243.7 | 220.4 | 23.3 |
| DFL clean | AB (A q + B c) | 357.8 | 296.7 | 261.7 | 35.0 |
| DFL gaps | R (v8) | 357.8 | 235.1 | 187.5 | 47.6 |
| DFL gaps | A (TCN) | 357.8 | 272.3 | 225.1 | 47.2 |
| DFL gaps | B (change detector) | 357.8 | 235.0 | 189.1 | 45.9 |
| DFL gaps | AB (A q + B c) | 357.8 | 281.1 | 228.5 | 52.6 |

## Spurious predicted possessions by kind (per match, mean over the 4 conventions)

| Setting | Arm | over_true_dead | over_other_team | merge_container | fragment_of_true | other |
|---|---|---:|---:|---:|---:|---:|
| SkillCorner real | R (v8) | 9.2 | 26.2 | 0.7 | 16.4 | 1.1 |
| SkillCorner real | A (TCN) | 5.0 | 23.8 | 0.8 | 15.4 | 0.8 |
| SkillCorner real | B (change detector) | 9.4 | 26.4 | 0.6 | 16.4 | 1.2 |
| SkillCorner real | AB (A q + B c) | 5.2 | 26.0 | 0.8 | 16.9 | 0.7 |
| DFL clean | R (v8) | 0.1 | 13.1 | 2.1 | 4.6 | 0.1 |
| DFL clean | A (TCN) | 0.0 | 19.8 | 1.0 | 9.9 | 0.0 |
| DFL clean | B (change detector) | 0.3 | 15.2 | 1.8 | 5.9 | 0.1 |
| DFL clean | AB (A q + B c) | 0.3 | 22.5 | 1.0 | 11.1 | 0.0 |
| DFL gaps | R (v8) | 12.2 | 20.3 | 2.5 | 10.9 | 1.7 |
| DFL gaps | A (TCN) | 7.9 | 23.1 | 1.6 | 13.8 | 0.7 |
| DFL gaps | B (change detector) | 11.3 | 19.6 | 2.6 | 10.6 | 1.7 |
| DFL gaps | AB (A q + B c) | 8.2 | 26.4 | 1.4 | 15.9 | 0.7 |

## Recall by true possession length (share matched at IoU >= 0.3)

| Setting | Truth | Arm | < 3 s | 3-6 s | 6-12 s | >= 12 s |
|---|---|---|---:|---:|---:|---:|
| SkillCorner real | raw | R (v8) | 11% | 43% | 66% | 85% |
| SkillCorner real | raw | A (TCN) | 12% | 45% | 66% | 86% |
| SkillCorner real | raw | B (change detector) | 11% | 44% | 67% | 85% |
| SkillCorner real | raw | AB (A q + B c) | 13% | 46% | 67% | 86% |
| SkillCorner real | controlled3 | R (v8) | 16% | 48% | 75% | 90% |
| SkillCorner real | controlled3 | A (TCN) | 19% | 51% | 76% | 90% |
| SkillCorner real | controlled3 | B (change detector) | 17% | 49% | 75% | 90% |
| SkillCorner real | controlled3 | AB (A q + B c) | 20% | 52% | 76% | 91% |
| DFL clean | raw | R (v8) | 17% | 55% | 81% | 94% |
| DFL clean | raw | A (TCN) | 31% | 72% | 89% | 97% |
| DFL clean | raw | B (change detector) | 19% | 58% | 82% | 95% |
| DFL clean | raw | AB (A q + B c) | 34% | 73% | 90% | 98% |
| DFL clean | controlled3 | R (v8) | 33% | 64% | 87% | 97% |
| DFL clean | controlled3 | A (TCN) | 48% | 81% | 93% | 98% |
| DFL clean | controlled3 | B (change detector) | 35% | 67% | 87% | 97% |
| DFL clean | controlled3 | AB (A q + B c) | 51% | 82% | 94% | 98% |
| DFL gaps | raw | R (v8) | 12% | 44% | 74% | 91% |
| DFL gaps | raw | A (TCN) | 22% | 58% | 84% | 96% |
| DFL gaps | raw | B (change detector) | 14% | 44% | 74% | 92% |
| DFL gaps | raw | AB (A q + B c) | 24% | 59% | 85% | 96% |
| DFL gaps | controlled3 | R (v8) | 23% | 50% | 77% | 94% |
| DFL gaps | controlled3 | A (TCN) | 34% | 67% | 86% | 98% |
| DFL gaps | controlled3 | B (change detector) | 24% | 50% | 77% | 95% |
| DFL gaps | controlled3 | AB (A q + B c) | 37% | 67% | 86% | 97% |

## Full-OOF decoder grid (selection score, for reference only; the gate uses the fold-wise choice)

- A (TCN): [1.0, 0.0] 0.5690; [1.0, 1.0] 0.5664; [0.5, 0.0] 0.5627; [1.0, 2.0] 0.5599; [0.5, 1.0] 0.5549
- B (change detector): [1.0, 3.0] 0.5069; [1.0, 4.0] 0.5061; [1.0, 2.0] 0.5047; [0.5, 2.0] 0.5034; [0.5, 1.0] 0.5034
- AB (A q + B c): [1.0, 0.0] 0.5690; [1.0, 1.0] 0.5683; [1.0, 2.0] 0.5667; [1.0, 3.0] 0.5645; [0.5, 0.0] 0.5627
- R (v8 setting only): 0.5031
