# EXPERIMENT-21: a richer decoder on the probabilities we already have

Registered 2026-11-12 17:27 IST (clock read with `date`), before any new decoder setting was scored against any
truth. The sha256 of this file is in `work/exp21/registration.sha256`. Code: `exp21/` (C kernel `viterbi.c`,
`decoder.py`, `evaluate.py`, `test_exp21.py`). Nothing in `v2/`, `v8/`, `v10/`, `exp18/`, `exp19/`, `exp20/`, `pff/` or
`bench/` is edited; they are imported read-only. No retraining for the study itself.

Code hashes at registration (sha256, first 16): decoder.py 7abdc03cc924eab9, viterbi.c 2977fc6e7ba22e49,
evaluate.py ced5883e63397608, test_exp21.py 39d4f4141083934d.

## Question

v8 decodes its per-frame probabilities with one switch penalty (0.5) for every transition. The top public entry
reports a Viterbi with minimum durations and separate costs per transition. Does a decoder of that family, tuned
only on external out-of-fold data, beat v8 as submitted (`Mastaadmi.csv`, bb0ba6f5) on PFF, without losing on
SkillCorner or DFL gaps? Applied to three probability sources: v8's own, arm A (v10) and arm P.

## Decoder family (fixed)

Energy of a path over one valid period: sum of per-frame costs -log(clip(q, 1e-6, 1)) / 25 (v8's units) plus
transition costs. States H, A, D after H (DH), D after A (DA); DH and DA both emit D. Parameters:

| name | meaning | v8 | v10 / arm P reference |
|---|---|---|---|
| cHA | H <-> A switch | 0.5 | 1.0 |
| cLD | live -> D | 0.5 | 0.5 |
| cDLs | D -> live, the team that had the ball before the stoppage restarts | 0.5 | 0.5 |
| cDLo | D -> live, the other team restarts | 0.5 | 0.5 |
| bD | dead-ball log-odds bias: D's frame cost becomes (-log q_D - bD) / 25 | 0 | 0 |
| Lp | minimum possession run, seconds (round(s x 25) frames, at least 1) | 0 | 0 |
| Ld | minimum dead-ball run, seconds | 0 | 0 |

Minimum durations are exact (expanded-state Viterbi: a run can only be left from its duration cap). The first and
the last run of each valid period are exempt (truncated by the period edge). A D run at the start of a period has no
known previous team, so leaving it costs min(cDLs, cDLo). Tie rule as `v2.methods.viterbi` and `exp18.decode`.

Exactness, tested before registration (`python -m exp21.test_exp21`, 7 of 7 pass): brute force over every path on
250 random short sequences with random costs and minimum durations (energy equal to the true constrained minimum,
1e-9); v8 special case equal to `v2.methods.viterbi` path for path on 40 random inputs; v10 special case equal to
`exp18.decode`; restart costs and bias move the path the expected way. Planted defects: switching from a non-cap
state, a reversed tie rule and a swapped restart cost each turn at least one test red (3 of 3 caught).

Harness dry run before registration (`python -m exp21.evaluate harness`, reference settings only, all already
spent): v8 at its own setting reproduces EXPERIMENT-18 R on 34 of 34 bench views (max abs spellF1 difference 0.0,
paths identical to `v2.methods.viterbi`); arm A and arm P at v10's setting reproduce EXPERIMENT-19 part 2 OOF rows
(0.0); on PFF, v8 and v10 reproduce EXPERIMENT-19 part 1 on 64 matches and arm P reproduces part 2 on PFF-test 32
(0.0). The same checks run again inside `oof` and `pff` and stop the run on any difference above 1e-9.

## Inputs (read-only)

| source | bench OOF (SkillCorner real 20, DFL clean 7, DFL gaps 7) | PFF |
|---|---|---|
| v8 | `bench.oof` q (EXPERIMENT-18 R) | `pff.views` q_v8, 64 matches |
| A | `external/exp18/arm_a/<view>.npz` q | `external/pff-cache/v10/pff-<m>.npz` q (frozen v10) |
| P | `external/exp19/arm_p/<view>.npz` q (fold models) | `external/exp19/arm_p/pff-<m>-real-eval.npz` (model T), PFF-test 32 |

The change head c of arms A and P is not used (v10's decoder has beta 0).

## Tuning (fixed)

Objective: mean of (SkillCorner real mean over matches) and (DFL gaps mean over matches) of mean-of-4 spellF1, the
official scorer under the four bench conventions (as EXPERIMENT-18 and 20). DFL clean is reported, never tuned on.

Coordinate search: parameters in the order cHA, cLD, cDLs, cDLo, bD, Lp, Ld. For each, every listed value is scored
with the others fixed; the best value is taken; ties keep the current value, else the earliest listed. A pass with no
change ends the search; at most 5 passes. Start: the source's reference decoder (v8 for v8; v10's for A and P).

| parameter | values |
|---|---|
| cHA | 0.25, 0.5, 0.75, 1, 1.5, 2, 3 |
| cLD, cDLs, cDLo | 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2 |
| bD | -2, -1, -0.5, 0, 0.5, 1, 2 |
| Lp (s) | 0, 0.2, 0.4, 0.6, 1, 1.5, 2 |
| Ld (s) | 0, 0.4, 0.8, 1.2, 2, 3, 5 |

Nested score: for each of the 5 bench folds (`exp18.views.fold_of`), the search runs on the real and gaps views of
the other 4 folds and its setting decodes the held-out fold's views (real, gaps and clean). The nested numbers in
the gate come only from these held-out decodes. The full search on all 27 real and gaps views fixes the one
setting per source used on PFF and on the competition match.

## Candidates and confirmation

| candidate | probabilities | confirmation set | baseline |
|---|---|---|---|
| V8N | v8 | PFF-64 (all 64 passed EXPERIMENT-19's instrument check) | v8 as submitted |
| AN | arm A (v10) | PFF-test 32 (`exp19.split.PFF_TEST`) | v8 as submitted, same 32 |
| PN | arm P | PFF-test 32 | v8 as submitted, same 32 |

## Gate (per candidate, every clause must hold; numbers against v8 as submitted unless stated)

- G1: PFF mean-of-4 spellF1 delta >= +0.015 on its confirmation set.
- G2: match wins >= 60% (39 of 64; 20 of 32).
- G3: PFF controlled3 delta >= 0.
- G4: SkillCorner real nested delta >= -0.005.
- G5: DFL gaps nested delta >= -0.005.
- G6 (added to the parent's suggested gate): PFF delta against the same probabilities with their own reference
  decoder >= 0, so a candidate cannot pass on its probabilities while the new decoder makes it worse. For V8N this
  is implied by G1.
- NO CHANGE: if the full search returns the reference setting, the candidate is not a new decoder and cannot pass.

Pick rule: among passing candidates, the highest lower bound of the match-level bootstrap 95% interval
(`exp19.common.bootstrap_ci`, seed 19, 10,000 resamples) of the delta against v8 on PFF-test 32, the population all
three share (V8N is scored on those 32 as a subset of its 64). Ties: V8N, then AN, then PN (least new machinery).
This replaces "the highest PFF lower bound" in the parent's brief because V8N's own confirmation set (64) differs
from AN's and PN's (32); comparing bounds over different populations is not like for like.

Outcomes: PASS (pick built and checked as below), FAIL (no candidate passes; v8 stays), harness defect (any reference
mismatch above 1e-9 stops the run; fixed and rerun before any number is read). There is no separate VOID branch for
the gate: per-match sd of decoder-level deltas on PFF was 0.0047 in EXPERIMENT-20, so the standard error of a
64-match mean is about 0.0006 (32 matches: 0.0009), and +0.015 is more than 15 standard errors; the run can see an
effect of the bar's size. Frames changed per match against the reference decoder are printed beside each verdict.

Reported, not gated: v8's own P 1.0 control (all four costs 1.0) on PFF-64, PFF-test 32 and the bench folds; each
candidate's reference decoder against v8; possession counts, recall of true possessions under 3 s, predicted
possessions over true dead time per match, predicted D share; DFL clean.

## Prior looks and trial count (disclosed)

- PFF-64 is not fresh for decoder questions. EXPERIMENT-19's diagnostic scored v8 at penalties 0.25, 0.5 and 1.0 on
  it (1.0 best: 0.586 against 0.573), and EXPERIMENT-20 scored a 0.75 penalty there (+0.0092, 53 of 64). This
  experiment exists partly because of those numbers. No PFF number enters tuning; the risk is a design chosen in the
  direction PFF already favours.
- Arm P's numbers against v8 at v10's decoder are spent: PFF-test +0.152 (32 of 32), SkillCorner real +0.017
  (18 of 20), DFL gaps +0.104 (7 of 7). PN is therefore expected to clear G1 to G5 on its probabilities alone; what is
  new for PN is G6 (does the new decoder help or hurt arm P) and the label-free match checks below. Arm A on PFF:
  v10 lost to v8 by -0.014 (EXPERIMENT-19 part 1).
- Trials: 3 confirmation tests (one setting per candidate), on top of the search, which is selection on external OOF
  data only. Project decoder trials so far: EXPERIMENT-18 15-setting grid, EXPERIMENT-20 6 MBR settings and 5
  penalties.

## Predictions (recorded to be checked)

1. V8N's full search raises cHA above 0.5 (expected 0.75 to 1.5).
2. V8N fails G1: PFF delta between +0.005 and +0.015 (the 1.0-penalty control gives about +0.013).
3. The search picks cDLo below cDLs (after a stoppage the other team restarts more often: throw-ins, goal kicks).
4. Lp stays at or below 0.4 s (v8 already misses short possessions; longer minimums cost recall).
5. Ld above 0 (dead-ball flicker is cheap to remove).
6. AN fails G1 (arm A's probabilities lose on PFF).
7. PN passes G1 to G5; G6 is a coin flip.
8. If PN passes and is picked, the quiet-stretch rule is the clause most likely to fail on the competition match.

## If a candidate is picked: build on the competition match (label-free only)

Local Mac only; data `../more-than-a-hack-2027/data`. V8N: v8's frozen pipeline produces q, the new decoder replaces
the final Viterbi. AN: frozen `v10/final.pt`. PN: train arm P's final model with the EXPERIMENT-19 `--model final`
recipe (27 external + 64 PFF), saved under `work/exp21-evaluation/` (wrapper in `exp21/`, `exp19/arm_p.py` unedited).
Checks, all on the frozen candidate CSV in `work/exp21-evaluation/`: two runs byte-identical; official validator;
possessions per minute and D share against v8 (reported); EXPERIMENT-19 amendments 2 and 3 via `exp19/adoption.py`
(fail if >= 10% more possessions per minute than v8 with at least half the extra starts inside quiet stretches, or
quiet-stretch starts >= 1.5 times v8's); tag agreement not below v8's minus 0.02 (v8 0.783, so >= 0.763). Any failed
check means the candidate is not offered for adoption. `Mastaadmi.csv` is not touched; adoption is the parent's.

Results go to `work/exp21/` (aggregates and per-match scores only) and `outputs/experiment-21-results.md`.
