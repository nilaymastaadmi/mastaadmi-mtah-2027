# EXPERIMENT-22: retrain the per-frame model with better data and longer context

Registered 12 November 2026 (17:25 IST), before any EXPERIMENT-22 outcome exists. `exp22/` does not exist yet; no
model has been trained, no Metrica label has been built and no score has been computed. Numbers seen so far are the
spent ones already in the repo: EXPERIMENT-18 R and arm A, EXPERIMENT-19 part 1 (v8 0.573 against v10 0.559 on PFF-64)
and part 2, EXPERIMENT-20, and v8's single public-window score (0.550, never used to select anything). Nothing here
reads competition frames, labels, tags or `evaluation/gt`; the competition match is touched only by the label-free
candidate build if the gate passes. No match data or per-frame output goes to a hosted model.

## Question

v8's per-frame evidence is a gradient-boosted model with a few frames of context. v10 (a conv net with about 4.4 s of
context) beat it out of fold but lost on 64 fresh PFF matches. Does one recipe that changes the training data and the
context length together (same-provider Metrica matches, about 25 s of context at 5 fps, team-swap augmentation,
short-possession weighting, SkillCorner down-weighted) and is blended with v8's own gradient-boosted probabilities
beat v8 on the fresh PFF matches and out of fold?

## Data

- SkillCorner open data, 20 matches (setting "real"); DFL/IDSSE, 7 matches (settings "clean" = `natural`, "gaps" =
  `contested`): exactly the EXPERIMENT-18 views (`external/exp18/views/`).
- Metrica Sample Games 1, 2, 3 (public, github.com/metrica-sports/sample-data, already in ignored
  `external/metrica-sample/`; nothing is downloaded). Same provider as the competition. Games 1 and 2: the CSV
  tracking and `RawEventsData.csv`; Game 3: the EPTS `tracking.txt` + `metadata.xml` and `events.json`. Coordinates
  are already normalised to [0, 1]; pitch 105 x 68 m (Game 3 metadata; the same assumed for Games 1 and 2). Home =
  the Home CSV for Games 1-2, `idLocalTeam` for Game 3. Game 3 player slots are one per player id across the
  metadata's DataFormatSpecification frame ranges (off-pitch = missing). Ball height is unknown (set to 0 for the
  degradation weights).
- PFF FC World Cup 2022, 64 matches (`external/pff-cache/views/`, EXPERIMENT-19 loader and truth, unchanged): the
  confirmation set. **Never trained on, never used for any choice.** All 64 passed EXPERIMENT-19's instrument checks.

## Metrica truth rule (`exp22/metrica.py`, fixed now, before any label is built)

Applied per period to the event list in time order. In-play events: PASS, CARRY, RECOVERY, SHOT, SET PIECE,
BALL LOST, BALL OUT, CHALLENGE. Frame numbers are the provider's (1-based, converted to 0-based indices).

1. **Dead spells (D).** (a) Before every SET PIECE: D from the End Frame of the last in-play event before it (other
   than the SET PIECE itself) to the SET PIECE's Start Frame (exclusive). (b) After every stoppage event: D from its
   frame to the Start Frame of the next in-play event. Stoppage events: BALL OUT (End Frame); SHOT whose subtype
   contains OUT or GOAL (End Frame); FAULT RECEIVED (Start Frame); BALL LOST whose subtype contains OFFSIDE, HAND BALL
   or END HALF (End Frame). D also runs from the last event's frame of a period to the period end if that last event
   is a stoppage; otherwise frames after the last event are X.
2. **Controlled possession (H/A) on every other frame.** The team of the most recent anchoring event: PASS, CARRY,
   RECOVERY, SHOT or SET PIECE, anchored at its Start Frame. BALL LOST and BALL OUT keep their team when that team is
   already in possession and otherwise change nothing (a touch without control). CHALLENGE never changes possession
   (duels and deflections without control do not count, per `starter-repo/CHALLENGE.md`). A loose ball stays with the
   last team in possession.
3. Frames before the first event of a period, frames outside the two periods, and frames not yet covered by an anchor
   are X (ungraded, never in a loss). Events whose team is missing anchor nothing.

**Instrument checks (`exp22.metrica check`, run before any model output exists; aggregates only):** per game,
nearest-team agreement with the H/A truth above 0.5 (`v2.experiment.mapping_check`), graded D share between 0.15
and 0.65, possessions (raw runs of H or A) per graded minute between 1.5 and 6, X share of in-period frames below 0.10.
A failure is a loader defect: fixed once and logged. A game that still fails is excluded before training and
disclosed. No exclusion after any model output is seen.

## Folds

The EXPERIMENT-18 folds, `v4.develop.fold_of` (5 match-grouped folds per dataset): SkillCorner and DFL unchanged, so
fold models compare with v8's stored R view by view. Metrica games get the same rule on their own sorted list
(Game 1 fold 0, Game 2 fold 1, Game 3 fold 2), so each Metrica game is predicted by a model that never saw it.

## Main recipe M (the only candidate eligible for adoption)

**Inputs.** The 90 EXPERIMENT-18 columns per 25 fps frame (72 v6 columns plus 18 geometry columns), built by the
existing code (`v3.features.build`, `v4.shape.team_shape`, `v6.control.control_features`, `exp18.geometry`).
Per-fold robust scaling as arm A (median and IQR from every 10th training frame, clip +-10, NaN to 0, a missing
indicator per column with any NaN in the fold's training frames). Then 5 fps: non-overlapping 5-frame blocks from
each half's first frame; each block is the mean of its frames' scaled values and indicators.

**Training views per fold.** DFL matches outside the fold under `natural`, `gaps`, `contested` (train purpose);
Metrica games outside the fold under the same three conditions (`v2.degrade.degrade`, the existing missing-ball
simulation, seeded by game, condition and purpose); SkillCorner matches outside the fold (real).

**Team-swap augmentation.** Every training view also enters mirrored: x -> 1 - x, y -> 1 - y for players and ball,
team indices swapped, labels H <-> A; features recomputed from the mirrored tracking with the same code.

**Targets and weights (per 5 fps block = its centre frame, offset 2).** Class H/A/D from the raw truth; X never
counts; SkillCorner class loss only where the ball is visible (as v8). Frame weight = view weight x boundary weight x
short-possession weight. View weight: SkillCorner 0.25, every other view 1. Boundary weight 2 within +-1 s of a change
event (H to A or A to H directly, as EXPERIMENT-18), else 1. Short-possession weight: 2 for frames inside a raw H or A
run shorter than 3 s, 1.5 for 3 to 6 s, else 1. Change head target: 1 for blocks within one block of a block that
contains a change event.

**Network.** 1x1 conv to 64 channels; 7 residual blocks (LayerNorm, kernel-3 conv with dilations 1, 2, 4, 8, 16, 32,
64, GELU, 1x1 conv, dropout 0.1); centred receptive field +-127 blocks (25.4 s each side). Heads: H/A/D logits and a
change logit, as arm A.

**Loss and optimiser.** Weighted cross-entropy on H/A/D plus 0.5 x binary cross-entropy of the change head (positive
weight 5, view weight applied, every non-X block). Chunks of 512 blocks inside one half, batch 16, AdamW lr 2e-3 with
cosine decay, weight decay 1e-4, 12 epochs, one epoch = training blocks / 512 chunks, chunk start sampled by a seeded
generator ([22, fold or 5 for the final model, epoch]), torch seed 22, deterministic algorithms, 3 threads. No early
stopping, no tuning on any held-out score.

**Output.** Per held-out 25 fps frame: q_conv and c by linear interpolation of the block outputs between block centres
(edge blocks held constant), q renormalised; uniform q and c = 0 outside play.

**Blend.** q = 0.5 q_conv + 0.5 q_gbm, where q_gbm is v8's own gradient-boosted evidence: out of fold the stored bench
OOF probabilities (`bench.oof`, the v6-recipe fold models inside v8) for SkillCorner and DFL; the frozen v6 model
(`v6.infer.load_frozen`, never trained on Metrica or PFF) for Metrica views and PFF (`q_v8` in the PFF cache). The
weight 0.5 is fixed now and never tuned.

**Decoders.** Gate: v8's decoder (`exp18.decode.decode(q, None, periods, 0.5, 0)`, identical to `v2.methods.blend`
weight 1 penalty 0.5). Also reported: EXPERIMENT-18's P 1.0 decoder (`decode(q, None, periods, 1.0, 0)`).

## Ablations (descriptive only, cannot be adopted)

- **A1 = conv alone:** M's q_conv without the blend (no extra training).
- **A2 = no Metrica:** M's recipe with Metrica removed from training (5 fold models + one model on the 27 matches),
  blended as M. Isolates the same-provider data.

## Reference and harness

- v8 out of fold = R: bench OOF q with v8's decoder. The harness must reproduce `work/exp18/result-R.json` per view
  and convention to 1e-9 before any recipe score is read, or it stops as a harness defect (fixed, then rerun).
- v8 on PFF = `q_v8` with v8's decoder; must reproduce `work/exp19/part1-rows.json` v8 per match to 1e-9, same rule.
- v8 on Metrica = frozen v6 q with v8's decoder (computed here; not a reproduction).
- Scorer: the official `starter-repo/evaluation/possession_metrics.py` under `raw`, `controlled1`, `controlled2`,
  `controlled3` (`v5/labels.py`), via `exp19.common.details`. Headline per match: mean of the four spellF1.

## Order and seal

1. Metrica loader and instrument checks; view caches (no model).
2. M folds 0-4 and A2 folds 0-4; OOF scores on SkillCorner, DFL and Metrica written to `work/exp22/oof.json`.
3. M final (SkillCorner 20 + DFL 7 + Metrica 3) and A2 final (27 matches), trained before PFF is read.
4. PFF-64 scored once. `exp22/evaluate.py pff` refuses to run unless `work/exp22/oof.json` and both final models
   exist; each run appends a line with a reason to `work/exp22/pff-unseal.log`. A rerun after a crash logs "resume".

## Gate (M against v8, v8's decoder; all must hold)

1. PFF-64 mean-of-4 spellF1 delta >= +0.015 (mean over 64 matches).
2. M wins (delta > 0) on at least 39 of 64 PFF matches.
3. PFF controlled3 spellF1 delta >= 0 (mean over matches).
4. SkillCorner real OOF mean-of-4 delta >= +0.010 (mean over 20).
5. DFL gaps OOF mean-of-4 delta >= -0.010 (mean over 7).

Reported beside the gate, not in it: DFL clean, Metrica natural and contested (each game separately), raw macroF1,
predicted against true possession counts, D shares, recall by true length (< 3 s, 3-6, 6-12, >= 12 s), the P 1.0
decoder, sign test and match bootstrap 95% interval, frozen v10 on PFF, and A1 and A2 on everything.

**VOID** (no verdict, the run could not have shown the effect) if M's v8-decoded output differs from v8's on fewer
than 250 graded frames for the median PFF match, or if two or more Metrica games are excluded by the instrument check.
Otherwise PASS if all five hold, else FAIL.

Trivial baseline: v8 itself (the incumbent), and frozen v10 on PFF (0.559, spent). A pass that does not also beat
v10 on PFF is still a pass against v8, reported as such.

## Stop rules

- Defects only: a crash, NaN or loader defect may be fixed once per component, logged with the reason. No change to
  architecture, features, weights, blend, decoder, folds or gate after any held-out score is seen.
- Futility: if M's fold-0 q_conv with v8's decoder has SkillCorner raw macroF1 more than 0.10 below R on fold 0's
  SkillCorner matches, M is reported as failed and no PFF score is computed.
- If training cannot finish by 15 November 06:00 IST, A2 is dropped first (reported as not run); M is never cut.
- A failed gate is final for this registration.

## Predictions (recorded to be checked, some expected wrong)

- P1: M beats v8 on SkillCorner OOF by +0.01 to +0.03.
- P2: M beats v8 on PFF-64 but by less than +0.015 (gate check 1 fails). Expected about +0.005 to +0.015.
- P3: A1 (conv alone) is below M on PFF (the blend helps on the fresh set).
- P4: A2 is within 0.005 of M on SkillCorner and PFF (Metrica's three games change little off-provider), and M beats
  A2 on Metrica OOF.
- P5: M predicts fewer possessions over true dead time on PFF than v10 did.

## If it passes

Final = M final (already trained on all 30 matches; no retraining). Candidate: q_conv from M final on the competition
inputs exactly as v10 (`v10.inputs`, v8's fused-ball path), blended 0.5 with v8's frozen v6 evidence on the same
inputs, v8's decoder. Label-free checks only: two runs with byte-identical CSVs; official validator; possessions per
minute and D share against v8; possession starts inside quiet stretches (`exp19/adoption.py` rules: fail if >= 10% more
possessions per minute than v8 with at least half the extra starts in quiet stretches, or quiet-stretch starts >= 1.5
times v8's); tag agreement (`v4.smart_tagging_audit --candidate`) not below v8 minus 0.02 (0.763). Frozen in
`work/exp22-evaluation/` with hashes. `Mastaadmi.csv` is not touched; adoption is the parent session's call.

## Limits stated now

- SkillCorner and DFL are spent for any fresh claim; their numbers are match-grouped cross-validation. PFF is the
  only fresh test, one tournament and one provider, broadcast-derived tracking (closer to SkillCorner than to the
  competition input).
- The Metrica truth is our rule over Metrica's events, a convention next to DFL's, SkillCorner's and PFF's; the
  competition truth is derived from corrected tracking, not events. Three games, so Metrica OOF is noisy.
- Recipe M changes several things at once; A1 and A2 isolate only the blend and Metrica.
- Checkpoint log: bridge workstream `mtah-exp22`.

## Amendment 1 (12 November, before any model is trained or scored; original sha256 67ab9bf1)

Clarification from the parent session, per the repository `CLAUDE.md`: Metrica Sample Game 1 was v1's training game and
Games 2 and 3 are spent test sets. Using all three as training data is allowed. Their out-of-fold numbers (settings
`met_clean`, `met_gaps`) are development and diagnostic only and are never cited as evidence for the gate or for
adoption; prediction P4's Metrica clause is a diagnostic. The gate stays exactly as registered: PFF-64 plus SkillCorner
and DFL out of fold. Nothing else changes.
