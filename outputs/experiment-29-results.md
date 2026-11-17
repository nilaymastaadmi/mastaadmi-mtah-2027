# EXPERIMENT-29 results: PASS

**Verdict: PASS.** Candidate B meets checks 1 to 3 (`work/exp29/gate.json`, registration sha256 4b4a8dda) and its build passes check 4 (`work/exp29-evaluation/freeze.json`, CSV sha256 ace6e2f7). Check 5 (PFF FC's terms) and adoption are the parent's and Nilay's.

Question: does adding the 64 PFF matches as a fourth training provider for v13's conv half help on a provider the model has never seen? Leave-one-provider-out: for each held-out provider, recipe M trained on the other original providers with PFF ("with", weight A 0.25 or B 1.0) and without it ("without"), each blended 50/50 with the stored out-of-fold boosted evidence and decoded as v13. No PFF score of a PFF-trained model was computed.

## Bar (with minus without)

| Check | A (0.25) | B (1.0) |
|---|---|---|
| 1a_sc_mean4>=+0.005 | +0.0292 pass | +0.0346 pass |
| 1b_sc_bootstrap95_lower>0 | +0.0220 pass | +0.0262 pass |
| 1c_sc_wins>=12/20 | 19 pass | 19 pass |
| 1d_sc_raw>=0 | +0.0221 pass | +0.0265 pass |
| 1e_sc_controlled3>=0 | +0.0359 pass | +0.0425 pass |
| 2a_dfl_gaps_mean4>=0 | +0.0290 pass | +0.0203 pass |
| 2b_dfl_gaps_wins>=4/7 | 7 pass | 7 pass |
| 2c_dfl_gaps_raw>=-0.005 | +0.0305 pass | +0.0239 pass |
| 3_pooled27_mean4>=+0.005 | +0.0292 pass | +0.0309 pass |

Harness: H1, recipe M through the EXPERIMENT-29 code with no PFF reproduces `v12/final.pt`: all 50 tensors, scaler and every zip record identical (one logged fix to the check script: the file hash is reported, not required, because torch.save names records after the file). H2, PFF rows as registered (128 PFF views, order and hold-outs asserted). H3, the arm-scoring code reproduces EXPERIMENT-27's v13 rows with EXPERIMENT-22's fold models (136/136 files, max difference 0.0).

## Scores

| Held out | Candidate | Mean-of-4 delta | Bootstrap 95% | Wins | Raw | Controlled3 | Median frames changed |
|---|---|---|---|---|---|---|---|
| SkillCorner (20) | A | +0.0292 | [+0.0220, +0.0359] | 19/20 | +0.0221 | +0.0359 | 4693 |
| SkillCorner (20) | B | +0.0346 | [+0.0262, +0.0422] | 19/20 | +0.0265 | +0.0425 | 5540 |
| DFL gaps (7) | A | +0.0290 | [+0.0178, +0.0392] | 7/7 | +0.0305 | +0.0277 | 6449 |
| DFL gaps (7) | B | +0.0203 | [+0.0107, +0.0312] | 7/7 | +0.0239 | +0.0202 | 6511 |
| DFL clean (7, reported only) | A | +0.0280 | [+0.0227, +0.0360] | 7/7 | +0.0389 | +0.0075 | 2949 |
| DFL clean (7, reported only) | B | +0.0301 | [+0.0229, +0.0374] | 7/7 | +0.0414 | +0.0087 | 3580 |

With minus without by label convention:

| Held out | Candidate | Raw | Controlled1 | Controlled2 | Controlled3 | Mean-of-4 |
|---|---|---|---|---|---|---|
| SkillCorner (20) | A | +0.0221 | +0.0268 | +0.0320 | +0.0359 | +0.0292 |
| SkillCorner (20) | B | +0.0265 | +0.0319 | +0.0375 | +0.0425 | +0.0346 |
| DFL gaps (7) | A | +0.0305 | +0.0315 | +0.0264 | +0.0277 | +0.0290 |
| DFL gaps (7) | B | +0.0239 | +0.0235 | +0.0135 | +0.0202 | +0.0203 |

Absolute mean-of-4 per arm, beside v13's stored out-of-fold rows (whose fold models saw other matches of the held-out provider):

| Held out | without | with A | with B | v13 out-of-fold |
|---|---|---|---|---|
| SkillCorner (20) | 0.4712 | 0.5004 | 0.5058 | 0.5286 |
| DFL gaps (7) | 0.5648 | 0.5938 | 0.5851 | 0.6152 |
| DFL clean (7, reported only) | 0.6462 | 0.6742 | 0.6763 | 0.7290 |

## Predictions (registered before any score)

- P1 (SkillCorner: both with arms above without, at least one by less than +0.005): A +0.0292, B +0.0346, wrong.
- P2 (B above A on the pooled mean-of-4): A +0.0292, B +0.0309, right.
- P3 (DFL gaps delta within -0.005 to +0.005 for both): A +0.0290, B +0.0203, wrong.
- P4 (each without arm below v13's out-of-fold mean-of-4 on its held-out provider): right.

Training times (two at a time, 3 threads each): H1 17 min, sc-without 10 min, dfl-without 10 min, sc-with-A 32 min, sc-with-B 32 min, dfl-with-A 30 min, dfl-with-B 30 min.

## Real-match build (check 4)

- two byte-identical full runs: ace6e2f7 / ace6e2f7 (needed equal) pass
- official validator (both runs): passed (needed passed) pass
- possession starts inside quiet stretches: 25 (needed < 27) pass
- Smart Tagging phase agreement: 0.7982 (needed >= 0.763) pass
- predicted D share: 0.482 (needed 0.283 to 0.542) pass
- median possession length (s): 10.02 (needed 3.0 to 10.12) pass
- possession starts: 201 (needed 190 to 270) pass
- frames that differ from v13 (`Mastaadmi.csv`, 045ed8a5): 6,931 of 199,319, all graded: first half 4,006, second half 2,925, pre-kickoff window 0

## Limits

- The leave-one-provider-out test stands in for an unseen-provider score; a final model trained on all four providers has none of its own.
- The "without" arms train on 10 matches (SkillCorner held out: DFL 7 + Metrica 3) and 23 (DFL held out: SkillCorner 20 + Metrica 3), so the measured gain is PFF added to a smaller base than the final model's; even the "with" arms stay below v13's out-of-fold scores, whose fold models saw other matches of the held-out provider. The final model adds PFF to 30 matches; its gain over v13 is expected to be smaller than +0.03 and is not measured.
- SkillCorner and DFL have informed many earlier decisions; here they are held-out providers for models that never saw them.
- On the match the build changes 6,931 frames of v13 and predicts fewer, longer possessions (201 starts against v13's 224). Two guards sit near their limits: median possession 10.02 s (limit 10.12) and 25 starts in quiet stretches (limit under 27).
- Outside review (Codex, read-only, code and aggregates only): the code before any arm score, then the gate and build code before the build; every finding fixed and re-checked; details in `decisions/log.md`.
