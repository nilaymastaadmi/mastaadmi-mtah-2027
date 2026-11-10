# EXPERIMENT-18: find the short possessions (turnover evidence for the decoder)

Registered 7 November 2026, before any EXPERIMENT-18 outcome exists. The only numbers seen are the bench's
(`outputs/bench-loss-decomposition.md`), which are the reference R below. Nothing here reads competition frames,
labels or tags; the competition match is touched only by the label-free `v10/` inference if an arm passes.

## Question

v8 matches only 11% to 17% of true possessions shorter than 3 s (raw labels). Most are absorbed by a long
prediction for the other team or merged into one of the same team. The oracle bound is +0.10 to +0.15 spellF1
under the 3 s controlled rule. A decoder search alone gave +0.005 (EXPERIMENT-08). Can new evidence of
turnovers, from a sequence model (arm A) or a dedicated change detector (arm B), recover part of that bound
out of fold?

## Data, folds and training convention (unchanged from the v8 recipe)

- SkillCorner open data, 20 matches, real broadcast tracking (setting "real").
- DFL/IDSSE, 7 matches: `natural` eval input (setting "clean"), `contested` eval input (setting "gaps", the
  EXPERIMENT-04 simulation of `v2/degrade.py`).
- Folds: `v4.develop.fold_of` (5 match-grouped folds, both datasets). Every held-out view is predicted by a
  model that never saw that match.
- Training views per fold, exactly as the v8 recipe (`v8.analogue.fold_model`): DFL matches outside the fold under
  the three `train`-purpose conditions (`natural`, `gaps`, `contested`); SkillCorner matches outside the fold.
  Truth for training is the `raw` provider label. Frames with label X never count in a loss. SkillCorner frames
  count in the class loss only where the ball is visible (as v8); they are always present as input context.
- Truth for scoring: the official `starter-repo/evaluation/possession_metrics.py`, under `raw`, `controlled1`,
  `controlled2` and `controlled3` (`v5/labels.py`). Headline: mean of the four.

## Reference R (v8 recipe)

The stored bench OOF probabilities (`external/bench-oof`, the v6-recipe fold models inside v8) decoded exactly as
v8 (`v2.methods.blend`, weight 1, penalty 0.5). The evaluator recomputes R and must reproduce the bench
`work/bench/results.json` spellF1 for every view and convention to 1e-9, or the run stops as a harness defect.

## Shared pieces (written once, used by both arms)

- `exp18/geometry.py`: per-frame ball-relative geometry, no labels, identities or clock. Ball visible flag; ball
  speed; ball speed change and direction change (cosine) between t-3 and t+3; for each team the nearest and
  second-nearest player distance to the ball and the nearest player's speed relative to the ball; home minus
  away nearest distance; players of each team within 5 m; the team of the nearest player and whether it differs
  from 5 frames earlier; ball distance to the touchline and goal line; ball off the pitch. Missing ball gives NaN.
- `exp18/views.py`: the view list, per-view inputs (v6 columns plus geometry), labels, periods, owner and the
  change targets, cached under ignored `external/exp18/`.
- Change event: frame t where the raw label goes directly from H to A or A to H (`labels[t-1]`, `labels[t]` both
  in {H, A} and different). Restarts after D are not change events (D handles them).
- `exp18/decode.py`: a change-aware Viterbi. Same states and costs as v8 (`-log(clip(q, 1e-6, 1)) / 25`). A switch
  into or out of D costs P_D = 0.5 as in v8. A switch H to A or A to H at frame t costs `P * exp(-beta * c_t)`,
  where c_t in [0, 1] is the arm's change evidence. beta = 0 and P = 0.5 is exactly the v8 decoder (a test
  checks equality with `v2.methods.viterbi`).
- Decoder grid (fixed now): P in {0.25, 0.5, 1.0} x beta in {0, 1, 2, 3, 4} (15 settings).
- Decoder selection is fold-wise, so the reported score is out of fold: for held-out fold f, the setting with the
  highest selection score on the other four folds' views is used on fold f. Selection score = mean of (SkillCorner
  real mean-of-4 spellF1 averaged over its matches) and (DFL gaps mean-of-4 averaged over its matches). Ties go to
  the earlier setting in grid order (P ascending, then beta ascending). DFL clean is never used for selection.

## Arm A: temporal sequence model (`exp18/arm_a.py`, torch in `.venv-video`)

- Input per frame: the 72 v6 columns plus the geometry columns. Per-fold robust scaling (median and IQR from that
  fold's training frames), clipped to +-10, NaN set to 0, plus one 0/1 missing indicator per column that has any
  NaN in that fold's training frames.
- Network: 1x1 conv to 64 channels; 7 residual blocks, each LayerNorm over channels, a kernel-3 dilated conv
  (dilations 1, 2, 4, 8, 16, 32, 48), GELU, a 1x1 conv and dropout 0.1. Receptive field +-111 frames (4.4 s),
  centred (offline). Two 1x1 heads: H/A/D logits and a change logit.
- Loss: frame cross-entropy on H/A/D, frames within +-25 frames of a change event weighted 2 (boundary-aware);
  plus 0.5 x binary cross-entropy of the change head against a target of 1 within +-6 frames of a change event,
  positive weight 5.
- Training: chunks of 2048 frames inside one half, drawn by a seeded generator; batch 16; AdamW, learning rate
  2e-3 with cosine decay, weight decay 1e-4; 12 epochs, one epoch = training frames / 2048 chunks; seed 18;
  CPU, `torch.use_deterministic_algorithms(True)`, fixed thread count. No early stopping, no OOF-based tuning.
- Output per held-out view: q (softmax H/A/D) and c (sigmoid change), stored under `external/exp18/arm_a/`.
- Decoder: `exp18/decode.py` with q and c and the fold-wise grid selection above.

## Arm B: possession-change detector (`exp18/arm_b.py`, `.venv`, built by a delegated Sonnet worker)

- Target per frame: 1 if a change event lies within +-12 frames (0.5 s).
- Features at frame t: the 72 v6 columns and the geometry columns at t; and for the 12 v6 control columns and the
  geometry columns, the differences x[t+d] - x[t-d] for d in {12, 25, 50} frames (NaN across a half boundary).
- Model: `HistGradientBoostingClassifier` binary, with the v8 recipe's `v2.experiment.HGB` parameters. Training
  rows: every 5th frame (fixed offset 0) of each training view, label not X; SkillCorner rows only where the ball
  is visible.
- Output per held-out view: c (probability per frame, every frame), stored under `external/exp18/arm_b/`.
- Decoder: `exp18/decode.py` with R's stored q (v8's own probabilities) and arm B's c, fold-wise grid selection.

## Blend AB (registered now, secondary)

Arm A's q with arm B's c, same decoder grid and selection. Eligible only if it passes the gate and beats the better
single passing arm on SkillCorner real by at least 0.005; otherwise the single arm is preferred.

## Gate (per arm, against R, per match, with the fold-wise selected decoder)

All of:

1. SkillCorner real: mean-of-4 spellF1 delta >= +0.02 (mean over 20 matches).
2. DFL gaps: mean-of-4 delta >= +0.02 (mean over 7 matches).
3. controlled3 delta >= 0 on SkillCorner real and on DFL gaps (means over matches).
4. Wins (mean-of-4 delta > 0) on at least 15 of 20 SkillCorner matches.
5. Wins on at least 5 of 7 DFL gaps matches.
6. Guard: DFL clean mean-of-4 delta >= -0.02.
7. Guard: SkillCorner real raw macroF1 delta >= -0.02.

If both arms pass, the one with the higher SkillCorner-real gain is preferred (then AB as above). Three candidates
are tested against one gate; this is disclosed, and the +0.02 and 15 of 20 thresholds are kept for that reason.

## Reported for every arm, not only spellF1

Per setting and convention: spellF1, macroF1 (raw), predicted against true possession count, recall by true length
class (< 3 s, 3-6, 6-12, >= 12 s) and the spurious counts, using `bench.decompose` and the bench length buckets.
Also the selected decoder setting per fold, and the per-match deltas.

## Stop rules

- Harness: R must reproduce the bench numbers (above), else stop and fix the harness before any arm is scored.
- Defects only: a code defect found on training loss, a NaN or a crash may be fixed once per arm, logged with the
  reason. No architecture, feature or loss change is made after seeing any held-out score.
- Futility (arm A): if the fold-0 model with the v8 decoder setting (P 0.5, beta 0) has SkillCorner raw macroF1 more
  than 0.10 below R on fold 0's SkillCorner matches, arm A stops and is reported as failed (no retry).
- A failed gate is final for this registration. A new idea needs a new registration.

## If an arm passes: v10

- Final model: the arm trained on all 27 external matches with the same recipe (same epochs, seed, parameters);
  decoder setting selected by the same selection score on the whole OOF set.
- `v10/infer.py`: label-free, Mac only, competition inputs exactly as v8 (`v8.infer.fused_inputs`), prints
  aggregates only. Run twice; the two CSV digests must be identical. Official validator must pass. Tag audit
  (`v4.smart_tagging_audit --candidate`) must reach at least 0.763 (v8 0.783 - 0.02).
- The CSV goes to `work/v10-evaluation/`, never to `Mastaadmi.csv`; the parent session decides adoption.

## Limits stated now

- External matches are spent for any fresh-holdout claim; this is match-grouped cross-validation.
- The selection score uses SkillCorner real and DFL gaps, the same settings the gate uses; the fold-wise selection
  keeps each match's decoder setting out of its own fold, but the grid itself is fixed by this registration.
- Raw provider labels count deflections; training on them may teach the model to call flips the competition does
  not count. The controlled3 gate condition is there for that reason.
- Checkpoint log: bridge workstream `mtah-exp18`.
