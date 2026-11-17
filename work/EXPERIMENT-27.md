# EXPERIMENT-27: test-time averaging over symmetric views with v12's frozen models

Registered 17 November 2026, 01:52 IST, before any EXPERIMENT-27 code, prediction or score exists. Requested by the parent session
with Nilay's yes; the bar is the parent's "EXPERIMENT-27 bar (rule v3 for EXP-27)" line dated 2026-11-17 in
`decisions/log.md`, copied below unchanged. No model is trained or tuned. Nothing here reads competition labels, tags
or `evaluation/gt`; the competition match is read label-free (view disagreement and possession counts, and the build
after a pass). No match data or per-frame output goes to a hosted model.

## Question

v12's models were trained with team-swap augmentation, so they should be roughly symmetric. Does averaging their
probabilities over four symmetric views of the same tracking (no retraining) beat v12?

## Views (tracking transformed, features recomputed, predictions mapped back)

- V0 identity.
- V1 touchline mirror: y -> 1 - y for players and ball, teams unchanged.
- V2 full swap: x -> 1 - x, y -> 1 - y, team indices swapped (`exp22.views.mirror`).
- V3 goal-line mirror with teams swapped: x -> 1 - x, team indices swapped, y unchanged.

Each view is applied to the same tracking the 90 EXPERIMENT-22 columns come from (`exp22.views.arrays`, degraded as
the view's condition, then transformed; PFF from `pff.load.load`; the match from v12's input path: tactical ATD with
the broadcast ball fused in). Features are recomputed from the transformed tracking with `exp22.views.features`
exactly as `exp22/views.py` does for swap views: the 72 v6 columns in float64 for the boosted half (as v8 runs) and the
90 columns cast to float32 for the conv half (as the training caches). For V2 and V3 the predicted H and A columns are
swapped back, so every view's probabilities refer to the original teams. The truth is always the original view's.

## Models (frozen; each out-of-fold prediction from models that never saw that match)

- Conv half: recipe M seed 22 (the network inside v12): fold models `external/exp22/M/models/fold<k>.pt` out of fold,
  the final model (sha256 e68c5767, = `v12/final.pt`) on PFF and the match. Three threads, as EXPERIMENT-22.
- Boosted half: v8's v6-recipe fold models `external/models/v6-recipe-fold<k>.pkl` (the models behind the stored
  bench OOF) out of fold on SkillCorner and DFL; the frozen v6 model (`v6.infer.load_frozen`) on Metrica, PFF and the
  match (as v12).
- Folds: `exp22.views.fold_of` (= `v4.develop.fold_of`), Metrica as its own folds.

## Candidates (at most 2) and decoder

- **T1:** q = 0.5 x mean over V0..V3 of q_conv + 0.5 x v12's boosted q (V0 only, untouched).
- **T2:** q = 0.5 x mean over V0..V3 of q_conv + 0.5 x mean over V0..V3 of q_boosted.
- Decoder: v12's, `exp18.decode.decode(q, None, periods, 0.5, 0)`. Weights fixed, never tuned.
- Reference: v12 = V0 of both halves (EXPERIMENT-22 M with v8's evidence).

## Harness (before any candidate score; a failure stops the run as a defect, one logged fix allowed)

- H1: on every out-of-fold view and every PFF match, V0 recomputed through this pipeline equals v12's stored
  probabilities: conv q byte-identical to `external/exp22/M/` (float32); boosted q equal to the stored bench OOF
  (compared after the same float32 cast it was stored with) on SkillCorner and DFL, to `q_gbm` on Metrica and `q_v8` on
  PFF. Then V0-only scoring reproduces EXPERIMENT-22's stored v12 rows (`work/exp22/oof-rows.json`,
  `pff-rows.json`) with max abs difference <= 1e-9.
- H2: bounds on every view's q (finite, non-negative, rows sum to 1 within 1e-5, length equals the view).
- H3: V2 recomputed on one DFL training view equals the cached EXPERIMENT-22 swap view's 90 columns exactly.

## Bar (copied unchanged from `decisions/log.md`, "EXPERIMENT-27 bar (rule v3 for EXP-27)")

> Bar vs v12, all must hold: PFF-64 mean-of-4 >= +0.003 with >= 36 of 64 wins, controlled3 >= 0 and raw >= 0;
> SkillCorner OOF mean-of-4 >= 0; DFL gaps mean-of-4 and raw each >= -0.005; all real-match guards of rule v2 clause
> (4). If both pass, the higher PFF-64 mean-of-4 wins. If adopted, the build also starts second-half features and
> decoding at the organiser kickoff (warm-up fix from the bug hunt) on top of the pre-kickoff fix.

Spelled out as checks per candidate (deltas are means over matches of candidate minus v12, v12's decoder, spellF1
from `exp19.common.details`; mean-of-4 = `exp19.common.mean4`):
1. PFF-64 mean-of-4 delta >= +0.003; 2. PFF-64 wins (mean-of-4 delta > 0) >= 36 of 64; 3. PFF-64 controlled3 delta
>= 0; 4. PFF-64 raw delta >= 0; 5. SkillCorner real OOF mean-of-4 delta >= 0 (20 matches); 6. DFL gaps OOF mean-of-4
delta >= -0.005 (7 matches); 7. DFL gaps OOF raw delta >= -0.005. Then the real-match guards of rule v2 clause (4) on
the built candidate: two byte-identical full runs, official validator, possession starts inside quiet stretches
< 27 (`exp19/adoption.py`), Smart Tagging phase agreement >= 0.763, predicted D share within 0.283 to 0.542, median
possession length within 3.0 to 10.12 s.

**VOID** if a harness check still fails after one logged fix. Otherwise PASS if a candidate meets checks 1 to 7 and
the guards, else FAIL. FAIL or VOID keeps v12 (with the pre-kickoff fix, `Mastaadmi.csv` 0cdc7351).

Reported, not decisive: DFL clean, Metrica (spent, diagnostic only), every convention, sign tests and bootstrap
intervals, frames changed against v12, each single view's score, and on the match (label-free): the share of graded
frames whose argmax differs between any two views (per half of the model and blended), and possessions per candidate.

## Order and seal

1. Registration hashed (`work/exp27/registration.sha256`). Code; H3.
2. Views and predictions for all out-of-fold views and PFF matches (`external/exp27/`); H1, H2.
3. Out-of-fold scores (`work/exp27/oof.json`). 4. PFF-64 scored once (refuses before `oof.json` exists; logged in
   `work/exp27/pff-unseal.log`, a rerun after a crash logs "resume"). 5. Bar applied mechanically
   (`work/exp27/gate.json`); label-free match check; results file, verdict first.
6. If a candidate passes: build on the match with the pre-kickoff fix (`v12/kickoff.py`) and the warm-up fix
   (second-half features, conv context and decoding start at the organiser kickoff frame 113,742; frames 110,808 to
   113,741 are D), twice, guards, freeze in `work/exp27-evaluation/`.

## Stop rules

- Defects only, fixed once per component and logged. No change to views, models, weights, decoder or bar after any
  held-out score is seen. Time: if no candidate is frozen by 17 Nov 10:00 IST, stop and report; v12 stays.

## Predictions (recorded to be checked; some expected wrong)

- P1: views disagree little: the median out-of-fold view changes fewer than 2% of graded frames between V0 and the
  4-view average for T1.
- P2: T1 changes PFF-64 mean-of-4 by -0.002 to +0.004 (check 1 likely fails).
- P3: T2 scores above T1 on PFF-64 mean-of-4.
- P4: V1 (touchline mirror, teams unchanged) is the view closest to V0.

## Limits stated now

- PFF-64 has judged at least 11 earlier candidates; SkillCorner and DFL are spent (match-grouped cross-validation).
- The warm-up fix in the build is not part of what the out-of-fold and PFF numbers measure (external matches have no
  pre-kickoff gap); it is the parent's decision, applied only if a candidate passes.
