# EXPERIMENT-29: PFF-64 as a fourth training provider for v13's conv half

Registered 17 November 2026, 14:55 IST, before any EXPERIMENT-29 code, prediction or score exists. Requested by the
parent session with Nilay's yes ("keep improving until the deadline"); the bar is the parent's "EXPERIMENT-29 bar
(rule v4)" line, the last line of `decisions/log.md` at commit 3c418e3, copied below unchanged. That line says it was
fixed "at 15:00 IST", but commit 3c418e3 is timestamped 14:50:40 IST, so the typed time is wrong; the order (bar, then
this registration, then code) is proved by commit 3c418e3 and the hash below, not by typed times. Nothing here reads
competition labels, tags or `evaluation/gt`; the competition match is read label-free, and only in the build after a
pass. No match data or per-frame output goes to a hosted model. No PFF score of a PFF-trained model is computed or
reported, and PFF is not used as a test anywhere in this experiment.

## Question

v13's conv half (recipe M of EXPERIMENT-22) was trained on 30 matches from three providers (SkillCorner 20, DFL 7,
Metrica 3). Does adding the 64 PFF FC World Cup matches as a fourth training provider help on a provider the model
has never seen? Evidence is leave-one-provider-out, because PFF stops being a holdout for the new model.

## Training recipe (EXPERIMENT-22 recipe M, unchanged)

Every model in this experiment is trained by `exp22.net.train` unchanged: architecture, 12 epochs, AdamW and cosine
schedule, chunk 512 blocks, batch 16, short-possession and boundary weights, change loss, `exp22.net.setup` (seed 22,
3 torch threads, deterministic algorithms), scaler fitted on that model's own training views (`fit_scaler`), and
seed index 5 (the final model's) for every training, so each "with" arm and its "without" arm share every seed.
`exp22/net.py` and `exp22/views.py` are not edited; EXPERIMENT-29 supplies the view list and a view loader for PFF.

## How PFF becomes training rows (fixed now)

- Frames: the existing PFF view cache `external/pff-cache/views/pff-<match>.npz` for all 64 matches (`pff.views`):
  25 fps (`pff.load` resamples PFF's about 29.97 fps tracking to 25 fps per period, labels by nearest frame), the
  same 90 feature columns (float32) as every other provider. No new feature is computed.
- Labels: the cache's `labels` field, PFF's raw-convention H/A/D/X truth from `pff/truth.py` (EXPERIMENT-19 rule).
- Team swap (part of recipe M: every training view enters unswapped and swapped): the swapped PFF view is
  EXPERIMENT-27's full-swap view `external/exp27/views/pff-<match>-V2.npz` (features recomputed from the mirrored
  tracking by `exp22.views.mirror` and `exp22.views.features`, the construction of every EXPERIMENT-22 swap view;
  EXPERIMENT-27 H3 showed it equals a cached EXPERIMENT-22 swap view exactly), with labels H and A exchanged and the
  same periods. So PFF contributes 128 training views (64 matches x 2), appended after the original views in match
  order, unswapped before swapped.
- Frames with no label: X frames and frames outside the halves get no class loss and no change-loss mask, exactly
  as `exp22.net.frame_targets` treats X for every provider; their features still feed the network's context. The
  class loss uses every labelled PFF frame (the DFL and Metrica rule); the visible-ball restriction stays
  SkillCorner-only, as in recipe M.
- Per-match weight w (A: 0.25, B: 1.0): applied as recipe M applies SkillCorner's 0.25 (`exp22.net.VIEW_W`): it
  multiplies every PFF frame's class-loss weight and change-loss mask inside the batch-normalised losses. Halves are
  still sampled in proportion to their length for every provider (unchanged), so w is a loss weight, not a sampling
  weight.

## Arms and candidates

- Held-out provider P = SkillCorner (20 matches) or DFL (7 matches, scored on its gaps condition; clean reported).
  "without": recipe M on all views of the other original providers (P = SkillCorner: DFL + Metrica; P = DFL:
  SkillCorner + Metrica). "with (w)": the same views plus the 128 PFF views at weight w. Six trainings: 2 without,
  4 with.
- Scoring each arm: its model on EXPERIMENT-27's four views of every held-out view (`external/exp27/views/`, H/A
  swapped back for V2 and V3, `exp22.net.predict`), q = 0.5 x mean over the four views of that q_conv + 0.5 x mean
  over the four views of the stored out-of-fold boosted q, built exactly as EXPERIMENT-27 builds T2 (V0's boosted q
  is the stored bench OOF), decoded with v12's decoder `exp18.decode.decode(q, None, periods, 0.5, 0)`. The only
  difference between arms is the conv model. The four-view averaging costs seconds per arm, so v13's pipeline is
  used (not the plain v12 pipeline).
- Candidates: A (w = 0.25) and B (w = 1.0). Each candidate's final model is recipe M on all 94 matches (the 100
  original views plus the 128 PFF views at its w); the candidate = v13 with the conv half's model replaced by that
  final model (its own scaler); boosted half, four-view averaging, blend, decoder, warm-up and pre-kickoff steps
  unchanged. The finals are trained while arms are scored; only the selected one is built.
- Leave-Metrica-out (diagnostic only) is not run: it would cost training time on the critical path.

## Harness (before any arm score; a failure stops the run as a defect, one logged fix allowed)

- H1 (the training path is the frozen recipe): recipe M through the EXPERIMENT-29 code with no PFF and nothing held
  out (`exp22.views.train_views(None, True)`, seed index 5) reproduces v12's final model: every state-dict tensor and
  the scaler arrays (med, iqr, nan_cols) byte-identical to `v12/final.pt` (sha256 e68c5767); the saved file's sha256
  reported beside it.
- H2 (PFF rows are what this registration says): for all 64 matches, EXPERIMENT-27's V0 x and periods equal the PFF
  cache's byte for byte and its V2 periods equal the cache's; the loader's unswapped PFF view returns the cache's x,
  labels and periods, and its swapped view returns V2's x with H and A exchanged in the labels (D and X unchanged).
- H3 (the scoring path is v13's): the arm-scoring code run with EXPERIMENT-22's stored fold models
  (`exp27.conv.model_for`) in place of an arm model reproduces EXPERIMENT-27's stored `T2|D1` rows on all 20
  SkillCorner and 14 DFL views, max abs difference <= 1e-9 over all conventions (spellF1 and macroF1).
- H4: bounds on every arm's q (finite, non-negative, rows sum to 1 within 1e-5 with rtol 0, length equals the view);
  every training's losses finite (`exp22.net.train` stops otherwise).

## Bar (copied unchanged from `decisions/log.md`, "EXPERIMENT-29 bar (rule v4)", commit 3c418e3)

> Bar per candidate, all must hold: (1) P = SkillCorner (20 matches), with minus without: mean-of-4 >= +0.005,
> bootstrap 95% lower bound > 0, >= 12 of 20 wins, raw >= 0, controlled3 >= 0; (2) P = DFL gaps (7 matches):
> mean-of-4 >= 0 with >= 4 of 7 wins, raw >= -0.005; (3) pooled over the 27 matches: mean-of-4 >= +0.005; (4) the
> final build passes all real-match guards of rule v2 clause (4) and predicts between 190 and 270 possession starts
> (v13: 224); (5) PFF FC's terms allow publishing a model trained on its data, checked and recorded before adoption.
> If both pass, the higher pooled mean-of-4 wins. Leave-Metrica-out is diagnostic only (spent games). Cut-offs:
> verdict by 19:00 IST, fully packaged by 22:30 IST, else v13 stays.

Spelled out per candidate (deltas are per-match "with (w)" minus "without" on the held-out provider, spellF1 from
`exp19.common.details`, mean-of-4 = `exp19.common.mean4`, paired statistics from `exp26.evaluate.paired` with
"without" as the base; wins = matches with mean-of-4 delta > 0):
1a. SkillCorner mean-of-4 delta >= +0.005; 1b. its bootstrap 95% lower bound > 0 (`exp19.common.bootstrap_ci`,
match-level, 10,000 resamples, seed 19); 1c. wins >= 12 of 20; 1d. raw delta >= 0; 1e. controlled3 delta >= 0.
2a. DFL gaps mean-of-4 delta >= 0; 2b. wins >= 4 of 7; 2c. raw delta >= -0.005.
3. Pooled: mean over the 27 matches (20 SkillCorner + 7 DFL gaps) of the mean-of-4 delta >= +0.005.
4. On the built candidate (its final model): two byte-identical full runs, official validator, possession starts
inside quiet stretches < 27 (`exp19/adoption.py`), Smart Tagging phase agreement >= 0.763
(`v4.smart_tagging_audit --candidate`), predicted D share within 0.283 to 0.542, median possession length within 3.0
to 10.12 s, and possession starts (`exp19.adoption.stats`, after the pre-kickoff step) between 190 and 270.
5. PFF FC's terms: checked by the parent; nothing is adopted without it.
Checks 1 to 3 are applied mechanically to both candidates; if both pass, the higher pooled mean-of-4 is selected and
only it is built.

**VOID** if a harness check still fails after one logged fix. **FAIL** if no candidate meets checks 1 to 3, or the
selected candidate's build fails check 4. **PASS** if a candidate meets checks 1 to 4 (adoption then also needs
check 5 and Nilay's yes). FAIL or VOID keeps v13 (`Mastaadmi.csv` 045ed8a5).

Reported, not decisive: DFL clean (7), every convention, sign tests, raw bootstrap intervals, frames changed, each
arm's absolute scores beside v13's stored out-of-fold T2 rows, training losses and times.

## Order

1. Registration hashed (`work/exp29/registration.sha256`). Code (`exp29/`); H2.
2. Trainings (H1's model, the 6 arms, then the 2 finals), at most two at a time (3 torch threads each), models in
   `external/exp29/models/`, logs in `work/exp29/`. H1 checked as soon as its model exists.
3. H3, then arm predictions and scores (`work/exp29/arms.json`, per-match rows `work/exp29/arm-rows.json`), only
   after H1 and H2 pass.
4. Bar applied mechanically (`work/exp29/gate.json`); one outside review (Codex, read-only, an isolated folder with
   code and aggregate JSON only) before the verdict is acted on; results file, verdict first.
5. If a candidate passes checks 1 to 3: build on the match exactly as v13 with its final model, twice, guards, freeze
   in `work/exp29-evaluation/`, then package as `v14/` (note and README say plainly that v14's model was trained on
   PFF, so it has no unseen-provider score of its own, and that this leave-one-provider-out test stands in for it).

## Stop rules

- Defects only, fixed once per component and logged. No change to the recipe, PFF rows, weights, arms, scoring or bar
  after any arm score is seen. Time: no verdict by 19:00 IST, or no complete v14 package by 22:30 IST: stop and
  report; v13 stays.

## Predictions (recorded to be checked; some expected wrong)

- P1: on SkillCorner, both "with" arms beat "without" on mean-of-4, but by less than +0.005 for at least one of them
  (check 1a fails for A or B).
- P2: B (w = 1.0) scores above A (w = 0.25) on the pooled mean-of-4.
- P3: on DFL gaps, "with" minus "without" lies between -0.005 and +0.005 for both candidates.
- P4: each "without" arm scores below v13's stored out-of-fold T2 on its held-out provider (mean-of-4), because the
  fold models behind T2 saw other matches of that provider.

## Limits stated now

- A leave-one-provider-out gain is evidence about an unseen provider, but the final model is trained on all four
  providers and has no unseen-provider score of its own; this test stands in for it.
- SkillCorner and DFL are spent (match-grouped cross-validation earlier); here they serve as held-out providers for
  models that never saw them, which is a different question from the earlier folds, but the same 27 matches have
  informed many earlier decisions.
- Two weights are tried (2 trials); the bar's thresholds were fixed by the parent before any score.
