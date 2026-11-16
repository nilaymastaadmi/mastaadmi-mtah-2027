# EXPERIMENT-19: PFF World Cup 2022 as a fresh test of v10, then as more training data

Registered 11 November 2026, before any EXPERIMENT-19 outcome exists. No model output on PFF has been computed or
read. Seen so far: the PFF specifications, field names and per-match aggregates of the raw data and of the truth rule
below on one match (10502: frame counts, event counts, D share 0.35, 309 possessions, nearest-team agreement 0.74),
used only to check that the loader works. Nothing here reads competition frames, labels, tags or `evaluation/gt`;
the competition match is touched only by label-free `v11/` inference if part 2 passes.

## Data

PFF FC World Cup 2022, 64 matches, in ignored `external/pff/` (260 of 260 files verified, `external/pff/VERIFY.txt`),
from the folder Nilay shared after requesting access through PFF's form. Licence: Nilay requested access; the terms
are recorded once he supplies them. No PFF data or derived per-frame array is published or sent to any hosted model
or worker; only aggregates leave the machine. Every model in this repository is untouched by these matches.

## Loader (`pff/load.py`, fixed now)

- Periods 1 and 2 only; extra time and shoot-outs excluded. One record per frameNum (PFF duplicates frames that carry
  several possession events; positions are identical).
- Players: raw `homePlayers` / `awayPlayers` (not the Kalman-smoothed copies); one slot per (team, jersey); a
  duplicated jersey within a frame keeps the higher confidence, then the first. Home team 0, away 1.
- Ball: the raw `balls` entry when exactly one is given, VISIBLE or ESTIMATED; none or several gives a missing ball.
  `ballsSmoothed` is never read, because PFF builds it from the event data (it would leak the truth).
- Coordinates: (x + L/2) / L, (y + W/2) / W with the metadata pitch length L and width W, as for SkillCorner.
- 25 fps resample per period from the first frame's time (PFF is 29.97 fps): positions linearly interpolated between
  the two neighbouring native frames when both are finite and at most 2 frameNums apart, labels from the nearest
  native frame. One ungraded separator frame between the halves, as for SkillCorner.
- Features (`pff/views.py`): exactly the SkillCorner real-tracking view of `exp18.views` (72 v6 columns plus 18
  geometry columns, float32), no degradation.

## Truth rule (`pff/truth.py`, fixed now, before any score)

From PFF's own game events, embedded in the tracking frames (`game_event`, Tracking spec v2.2 section 5), collected
by `game_event_id` with their `start_frame`, applied in time order:

- OTB (player on the ball) and the kick-offs: team in possession = the sequence-level `home_ball` (H if true, else A).
  PFF defines a sequence as an uninterrupted possession by one team, so an opponent touch that does not end the
  sequence does not flip the label (closest available to the competition's controlled-possession definition). If
  `home_ball` is missing, `home_team` is used; if both are missing, the state is unchanged.
- OUT (ball out of play, including whistles and goals) and END: D until the next OTB or kick-off.
- Before the first event of a period: D. Inside a VID (video missing) event: X (not graded).
- Other events (SUB, ON, OFF, G, FOUL) change nothing.

Scoring: the official `starter-repo/evaluation/possession_metrics.py` under the bench conventions `raw`,
`controlled1`, `controlled2`, `controlled3` (`v5/labels.py`). Headline per match: the mean of the four spellF1.

Instrument checks (harness, run before any model output is read; `exp19.part1 check`): per match, nearest-team
agreement with the H/A truth above 0.5 (`v2.experiment.mapping_check`), graded D share between 0.15 and 0.65, at
least 2 OUT events. A failure is treated as a loader defect: fixed once and logged. A match that still fails is
excluded before scoring and disclosed. No exclusion is made after any model output is seen.

## Part 1: fresh test of v10 against the v8 recipe (all 64 matches)

- v8 recipe: the frozen v6 model inside v8 (`work/v6-evaluation/model.pkl`, loaded with its hash and runtime checks)
  decoded as v8 (`v2.methods.blend`, weight 1, penalty 0.5, prior50). PFF has one tracking source, so there is no
  broadcast fusion step.
- v10: frozen `v10/final.pt` (sha256 checked against `work/v10-evaluation/freeze.json`) and its decoder at the frozen
  setting (P 1.0, beta 0), unchanged.
- Reported: mean-of-4 spellF1 for both and the per-match delta (v10 minus v8), wins/losses, a two-sided sign test and a
  match bootstrap 95% interval (descriptive); per convention spellF1, raw macroF1, predicted against true possession
  counts, D shares and recall of true possessions by length (< 3 s, 3-6, 6-12, >= 12 s).
- **Confirmation (registered):** v10 is confirmed better than v8 on PFF if all of: mean-of-4 delta > 0; v10 wins on at
  least 39 of 64 matches (60%); controlled3 delta >= 0. **Contradicted** if the mean delta < 0 and v10 wins on at most
  32 of 64. Anything else: **inconclusive**. A loss is a valid result and changes no file.

## Part 2: retrain arm A with PFF matches (arm P)

- Split (fixed now, `exp19/split.py`): the 64 ids sorted ascending; even positions are PFF-train (32), odd positions
  PFF-test (32). Each half holds 24 group-stage and 8 knockout matches.
  - PFF-train: 3812 3814 3816 3818 3820 3822 3824 3826 3828 3830 3832 3834 3836 3838 3840 3842 3844 3846 3848 3850
    3852 3854 3856 3858 10502 10504 10506 10508 10510 10512 10514 10516.
  - PFF-test: 3813 3815 3817 3819 3821 3823 3825 3827 3829 3831 3833 3835 3837 3839 3841 3843 3845 3847 3849 3851
    3853 3855 3857 3859 10503 10505 10507 10509 10511 10513 10515 10517.
- Recipe (`exp19/arm_p.py`): EXPERIMENT-18 arm A unchanged (network, losses, chunks, batch, optimiser, 12 epochs with
  one epoch = training frames / 2048 chunks, seed 18, per-epoch seeds [18, index, epoch], scaler on every 10th
  training frame), imported from `exp18/arm_a.py`. PFF views are treated as real broadcast tracking: class loss only on
  visible-ball frames, as SkillCorner; truth for training is the `raw` PFF rule above. Only the training view list
  and the thread count (5, so two runs can share the machine; arm A used 6) differ.
- Models:
  - **T**: the 27 external matches (exactly v10's final training views) plus PFF-train; seeds index 5, as v10's final
    model. So T differs from v10 only by the PFF-train data (and the thread count).
  - **Folds 0-4**: the EXPERIMENT-18 fold training views plus PFF-train; seeds index = fold, as arm A.
- Decoder: fixed at v10's setting (P 1.0, beta 0) for every arm and model. No re-selection (this also avoids the
  nested-selection issue disclosed for EXPERIMENT-18).
- Comparisons:
  - PFF held-out: T against the frozen v10 on the 32 PFF-test matches.
  - SkillCorner real (20), DFL clean and DFL gaps (7 each): fold models out of fold against the stored EXPERIMENT-18
    arm A out-of-fold probabilities (`external/exp18/arm_a/`), both decoded at (1.0, 0).
- **Gate (all must hold):**
  1. PFF-test mean-of-4 delta >= +0.02 (mean over 32 matches).
  2. SkillCorner real mean-of-4 delta >= +0.02 (mean over 20).
  3. Wins on at least 21 of 32 PFF-test matches.
  4. Wins on at least 13 of 20 SkillCorner matches.
  5. controlled3 delta >= 0 on PFF-test and on SkillCorner real.
  6. Guards: DFL gaps and DFL clean mean-of-4 delta >= -0.02 each.
  7. Guard: SkillCorner real raw macroF1 delta >= -0.02.
- Order and stop rule: T is trained first. If checks 1, 3 or the PFF half of 5 fail, the gate has failed; the folds
  are then run only as diagnostics if time allows, and the verdict stays FAIL.
- Defects only: a crash, NaN or loader defect may be fixed once, logged with the reason. No change to architecture,
  features, loss, split, decoder or gate after any held-out score is seen. A failed gate is final for this
  registration.

## If part 2 passes: v11

- Final model: arm P recipe on the 27 external matches plus all 64 PFF matches, seeds index 5, decoder (1.0, 0).
- `v11/` like `v10/`: competition inputs exactly as v10 (`v10.inputs.competition_x`, label-free, Mac only), frozen and
  hash-bound. Run twice: the two CSV digests must be identical. Official validator must pass. Tag audit
  (`v4.smart_tagging_audit --candidate`) must reach at least 0.763.
- Portability check as EXPERIMENT-18's on the public views (Mac side and a 1-thread against 5-thread rerun here; the
  Linux run needs a push, which is the parent's call).
- The CSV goes to `work/v11-evaluation/` only. Adoption is the parent session's.

## Limits stated now

- The PFF truth is our rule over PFF's events, a fourth convention next to DFL and SkillCorner. Training on it and
  testing on it favours arm P on PFF-test (convention learning); the SkillCorner condition is there for that reason.
- PFF tracking is broadcast-derived with estimated off-screen players and an estimated ball in parts; about a quarter
  of frames have no ball. The competition match is tactical tracking with a fused broadcast ball, so PFF is closer to
  SkillCorner than to the competition input.
- Part 1 tests 64 matches from one tournament and one provider; it is independent of every model, but not of the
  truth convention choice.
- Checkpoint log: bridge workstream `mtah-exp19`.

## Amendment 1 (11 November, before any score; Fable read-only review of the code)

No model output on PFF existed when these were made; the partial loader cache (17 matches) was deleted and rebuilt.

- Defect fixed: the resampler's exact-hit test used `np.isclose` with its default relative tolerance, which grew with
  time in the half, so late frames snapped to the next native frame instead of interpolating. Now absolute 1e-9.
- Game events are read from every record, including the duplicated frames the loader drops (a second event starting
  on a duplicated frame was lost). An event belongs to the half of the first frame carrying it; a start frame before
  that half's first tracked frame applies from the first frame (counted).
- Loaded coordinates are float32-rounded on both the first build and cached loads, so views do not depend on cache
  state.
- Instrument check: D share is over graded (non-X) frames. Matches failing it are excluded from scoring by the code
  (as registered). Thresholds scale with the included count: part 1 confirmation needs ceil(0.6 n) wins (39 of 64),
  contradiction at most floor(n / 2); part 2 PFF-test wins need ceil(21/32 n) of the included PFF-test matches.
- Disclosure for part 2: the scaler's missing-indicator columns are refit with PFF frames, so T's input width and
  seeded initialisation can differ from v10's even with the same seeds.

## Amendment 2 (11 November, before any SkillCorner, DFL or competition-match number): parent's adoption checks

These are not gate checks; the gate above is unchanged. They decide adoption on top of a passed gate. Fixed now, before
the arm P fold results or any v11 output exist (only the PFF-test numbers of model T are known).

1. v11 minus v8 (mean-of-4, wins, controlled3) on PFF-test and on SkillCorner real out of fold, and per match on DFL
   gaps, reported next to the gate. v8 on PFF-test is the frozen v6 model with the v8 decoder; out of fold it is
   EXPERIMENT-18's R (bench probabilities, v8 decoder). v11 must beat v8 on PFF-test and SkillCorner real.
2. Label-free on the competition match (`exp19/adoption.py`), for v8 (the current Mastaadmi.csv, sha256 bb0ba6f5) and
   v11: predicted possessions per graded minute, predicted D share, and the share of possession starts that fall inside
   a quiet stretch.
   - Possession start: a graded frame whose predicted label is H or A and differs from the previous frame's label in
     the same half.
   - Quiet stretch: a maximal run, inside one half, of frames with no ball in the model's input (the fused ball, as v10
     and v11 see it) lasting at least 5 s (125 frames), whose mean player speed over the run is below 1.5 m/s. Player
     speed: per frame, the mean over tracked players of the centred 5-frame speed of the tactical tracking.
   - **Fail for adoption** if v11 has at least 10% more possessions per minute than v8 and at least half of the extra
     possession starts (v11 starts minus v8 starts) fall inside quiet stretches.

## Amendment 3 (11 November, before any v11 SkillCorner, DFL or competition-match number): stricter adoption guard

Requested by the parent after the label-free dry run of `exp19/adoption.py` with v10 in the candidate slot (v8 18
possession starts inside quiet stretches, v10 36; v10 would have passed the amendment 2 rule). Disclosed: this
threshold was set after seeing that v8 and v10 count on the match, which is label-free; nothing about v11 is known.
The gate is unchanged.

- **Adoption also fails** if v11's possession starts inside quiet stretches number at least 1.5 times v8's (v8: 18,
  so 27 or more fails), whatever its possessions per minute. Either this rule or the amendment 2 rule fails adoption.
- v11's numbers are reported next to v8's and v10's.
