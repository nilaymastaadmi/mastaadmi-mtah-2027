# EXPERIMENT-19 results: PFF World Cup 2022 as a fresh test, then as training data

Registration: `work/EXPERIMENT-19.md`, written 11 November before any PFF model output (sha256 449475ac), with
amendment 1 from a Fable read-only code review, also before any score (sha256 0c1df78b), and amendments 2 and 3 (the
parent's adoption checks, before any SkillCorner, DFL or match number for arm P; final sha256 4c17db49). Code: `pff/` (loader, label
rule, views, v10 probabilities, tests) and `exp19/` (scoring, split, arm P training, diagnostics). Data: 64 PFF FC
matches in ignored `external/pff/`; nothing derived from them per frame is published or sent to any hosted model.
Licence terms not yet supplied by Nilay, so not recorded.

## Part 1 verdict: CONTRADICTED. v10 loses to v8 on the 64 PFF matches

| | v8 recipe | v10 (submission until 11 Nov) | v10 minus v8 |
|---|---:|---:|---:|
| Mean-of-4 spellF1 (64 matches) | 0.573 | 0.559 | **-0.014** |
| Matches won | 45 | 19 | |
| raw / controlled1 / controlled2 / controlled3 | 0.542 / 0.554 / 0.589 / 0.606 | 0.532 / 0.543 / 0.574 / 0.587 | -0.010 / -0.011 / -0.015 / -0.019 |
| Raw macroF1 | 0.856 | 0.854 | -0.002 |

- The registered rule needed a positive mean delta, at least 39 of 64 wins and a controlled3 delta of 0 or more. v10
  has a negative delta, 19 wins and a controlled3 delta of -0.019, which meets the registered "contradicted" line
  (negative delta and at most 32 wins).
- Sign test p = 0.0016 (19 against 45). Match bootstrap 95% interval of the delta: -0.020 to -0.008. Per-match deltas
  run from -0.082 to +0.045, median -0.015. Group stage -0.014 (15 of 48 wins), knockout -0.015 (4 of 16).
- All 64 matches passed the instrument checks (nearest-team agreement with the truth at least 0.64, graded D share
  0.28 to 0.52), so none was excluded.

### What v10 does differently on PFF

| Per match, raw truth (278 true possessions) | v8 | v10 |
|---|---:|---:|
| Predicted possessions | 240 | 257 |
| Matched true possessions | 172.4 | 174.2 |
| Recall of true possessions under 3 s / 3-6 s / 6-12 s / 12 s or more | 16% / 51% / 78% / 91% | 19% / 52% / 78% / 91% |
| Predicted D share (true 0.411) | 0.423 | 0.416 |

v10 finds slightly more short possessions, as on DFL, but it also predicts about 17 more possessions per match that
match nothing, so spellF1 falls. On SkillCorner (EXPERIMENT-18) the same model won mostly by predicting fewer
spurious possessions; on PFF it predicts more.

### Diagnostic outside the verdict: is it the decoder penalty? No

`exp19/diag_part1.py`, `work/exp19/diag-part1.json`. Selects nothing.

| Mean-of-4 spellF1 at switch penalty | P 0.25 | P 0.5 | P 1.0 |
|---|---:|---:|---:|
| v8's probabilities (v8 decoder form) | 0.548 | 0.573 (v8 as registered) | 0.586 |
| v10's probabilities (beta 0) | 0.501 | 0.536 | 0.559 (v10 as registered) |

At every penalty v8's frame probabilities beat v10's, by 0.027 to 0.047. At P 1.0 v10 makes about twice v8's spurious
possessions over true dead time (39.7 against 19.2 per match) and over the other team (24.6 against 12.5). So v10's
learned frame evidence transfers worse to PFF's broadcast tracking than v8's gradient-boosted model does. A stiffer
penalty helps both models, and helps v8 more.

### Diagnostic outside the verdict: are v10's dead-time "spurious" possessions real play the truth misses?

Parent's request (Fable's one flip condition). `exp19/diag_dead.py`, `work/exp19/diag-dead.json`. PFF tracking has no
ball-status field, so this uses the raw tracking ball only, never the events. Registered decoder settings, raw truth.

| Frames of predicted possessions over true dead time | v8 | v10 | all true D frames | all true H/A frames |
|---|---:|---:|---:|---:|
| Spans per match (of which within 1 s of live truth) | 29.3 (8.4) | 39.7 (9.5) | | |
| Frames per match | 2,907 | 3,229 | 62,778 | 89,266 |
| Ball more than 1 m outside the pitch ("plainly out of play") | 0.1% | 0.8% | 4.6% | 0.5% |
| No ball | 36% | 31% | 70% | 6% |
| Ball slower than 2 m/s | 15% | 13% | 14% | 13% |
| Within 2 s of a camera-cut proxy (players jump over 1.5 m per frame) | 3.0% | 2.4% | 0.3% | 0.0% |

- Almost none of these frames are plainly out of play by ball position (v10 0.8%, v8 0.1%). About two thirds have a
  visible ball inside the pitch, so tracking alone cannot say they are dead; it cannot say they are live play either.
- v10's excess over v8 (10.4 spans per match) is mostly inside stoppages: 9.3 of the 10.4 are more than 1 s from any
  live frame, so they are not a timing disagreement at the edges of a stoppage. The camera-cut proxy is enriched about
  10 times in these spans but covers only 2 to 4% of their frames, so broadcast replays are neither confirmed nor
  ruled out as the source.
- **The decisive number:** if every dead-time false possession of both models is forgiven (set to D, so it costs
  nothing), v8 rises to 0.609 and v10 to 0.606. v10 still trails, by 0.003, winning 29 of 64. These spans explain about
  three quarters of the gap, but even granting the flip condition in full gives at best a tie, not a v10 win. (Forgiving
  is not crediting: if those spans were real possessions, a model would also need the right team and timing to match
  them, which this does not measure.)

### How to read this

- This is the first test of v10 on matches no model has seen, and it goes against v10. It is one provider (PFF,
  broadcast tracking with estimated off-screen players and an estimated ball; 68% of frames have a ball) and one truth
  convention (ours, from PFF's events: sequence-level team, ball out of play until the restart).
- The competition input is tactical tracking with a fused broadcast ball, which is neither SkillCorner nor PFF. v10
  won on SkillCorner real tracking (20 of 20) and loses here (19 of 64), so its edge depends on the tracking source.
  Which side the competition match falls on is not measured.
- Whether to go back to v8 is the parent's call. The evidence for v10 over v8 is now one out-of-fold source for it
  (SkillCorner, +0.026, plus DFL) and one fresh source against it (PFF, -0.014).

## Part 2 verdict: FAIL. No v11

Arm P (EXPERIMENT-18 arm A recipe with 32 PFF matches added to training) gains hugely on held-out PFF but loses a
little to arm A on SkillCorner and DFL, so the registered gate fails on checks 2, 4 and 5. A failed gate is final for
this registration: no final model was trained and no `v11` CSV exists. Decoder fixed at (P 1.0, beta 0) throughout.
`work/exp19/gate-P.json`, `work/exp19/part2-{pff,oof}-rows.json`.

| Gate check | Result | Pass |
|---|---|---|
| 1. PFF-test mean-of-4 delta (T against frozen v10) >= +0.02 | +0.169 | yes |
| 2. SkillCorner real delta (arm P folds against arm A OOF) >= +0.02 | **-0.008** (95% -0.015 to -0.001) | **no** |
| 3. PFF-test wins >= 21 of 32 | 32 of 32 | yes |
| 4. SkillCorner wins >= 13 of 20 | **8 of 20** | **no** |
| 5. controlled3 >= 0 on PFF-test and SkillCorner | +0.160 / **-0.008** | **no** |
| 6. Guards: DFL gaps and clean >= -0.02 | -0.002 / -0.012 | yes |
| 7. Guard: SkillCorner raw macroF1 >= -0.02 | -0.001 | yes |

| Mean-of-4 spellF1 | v8 | arm A / v10 | arm P | P minus A / v10 | P minus v8 |
|---|---:|---:|---:|---:|---:|
| PFF-test (32) | 0.569 | 0.553 | 0.722 | +0.169 (32 of 32) | +0.152 (32 of 32) |
| SkillCorner real OOF (20) | 0.512 | 0.538 | 0.530 | -0.008 (8 of 20) | +0.017 (18 of 20; 95% +0.010 to +0.024) |
| DFL gaps OOF (7) | 0.494 | 0.600 | 0.598 | -0.002 (2 of 7) | +0.104 (7 of 7) |
| DFL clean OOF (7) | 0.627 | 0.716 | 0.705 | -0.012 (0 of 7) | +0.077 (7 of 7) |

Per match on DFL gaps, P minus v8: +0.075, +0.125, +0.109, +0.115, +0.105, +0.100, +0.100 (J03WMX, J03WN1, J03WOH,
J03WOY, J03WPY, J03WQQ, J03WR9); P minus A: -0.005, -0.011, -0.003, +0.014, -0.007, -0.011, +0.008.
controlled3, P minus v8: PFF +0.135, SkillCorner +0.019, gaps +0.066.

What it means:
- PFF data teaches the model PFF's own stoppages and our truth convention (dead-time false possessions fall from 42.5
  to 7.1 per match on PFF-test) without helping on other providers: on SkillCorner it is a small, consistent loss
  against arm A (recall by length unchanged, under-3 s 12.3% against 11.6%).
- Against v8, the submission since 11 Nov, arm P is ahead on all three external sources, out of fold or held out. But
  the gate was registered against v10/arm A and failed, and these v8 comparisons were seen after the fact, so they are
  not an adoption case under this registration. A v8-relative test of arm P would need a new registration, and its
  SkillCorner, DFL and PFF-test numbers are now spent; only the label-free match checks (tag audit, amendment 2 and 3
  guards) would still be fresh.
- The amendment 2 and 3 match checks were not run for arm P, since no v11 exists. The v10 dry run of the check is in
  `work/exp19/adoption-match-v10-dryrun.json` (v8 18 starts in quiet stretches, v10 36).

### Part 2 detail: model T on the PFF-test half


Model T: the 27 external matches plus the 32 PFF-train matches, arm A recipe, sha256 df81ed57…, 5,903 s to train.
Decoder fixed at v10's (P 1.0, beta 0). `work/exp19/part2-pff-rows.json`.

| 32 PFF-test matches | v8 | v10 (frozen) | T | T minus v10 | T minus v8 |
|---|---:|---:|---:|---:|---:|
| Mean-of-4 spellF1 | 0.569 | 0.553 | 0.722 | **+0.169** (32 of 32) | **+0.152** (32 of 32) |
| controlled3 | | | | +0.160 | +0.135 |
| Raw macroF1 | 0.860 | 0.856 | 0.910 | +0.054 | +0.050 |
| Bootstrap 95% of the delta | | | | +0.157 to +0.181 | +0.138 to +0.166 |

Gate checks 1 (+0.02 on PFF-test), 3 (21 of 32 wins) and the PFF half of 5 (controlled3 >= 0) pass by a wide margin.
The parent's extra bar (beat v8 on PFF-test) also holds.

Where it comes from (per match, raw truth): predicted possessions over true dead time fall from 42.5 (v10) and
32.3 (v8) to 7.1; over the other team from 25.9 and 19.4 to 10.8; matched true possessions rise from 174 and 172 to
200. Under-3 s recall rises from 20% to 32%. Predicted D share 0.435 against a true 0.420.

**Caution.** A gain this large on PFF's own convention is mostly the model learning what a PFF stoppage looks like
in broadcast tracking (no ball on 70% of true dead frames against 6% of live ones; replays) and our event-derived
truth. The competition input is tactical tracking with a different ball-missing pattern (37.6% missing for other
reasons). The SkillCorner out-of-fold check was the guard against that, and it failed.

Training: model T 5,903 s; folds 0-4 4,031 to 5,722 s each, two lanes of 5 threads with lock files (fold 3 was
skipped by the second lane and trained once). Model sha256 prefixes: T df81ed57, folds 07e3bfc3, c9a0b704, d5e595ad,
8922c016, 07e982ae.

## Files written (all new) and allowlist lines needed

- Code: `pff/__init__.py`, `pff/load.py`, `pff/truth.py`, `pff/views.py`, `pff/score_v10.py`, `pff/test_pff.py`;
  `exp19/__init__.py`, `exp19/common.py`, `exp19/split.py`, `exp19/part1.py`, `exp19/part2.py`, `exp19/arm_p.py`,
  `exp19/diag_part1.py`, `exp19/diag_dead.py`, `exp19/adoption.py`.
- Registration and results: `work/EXPERIMENT-19.md`, `outputs/experiment-19-results.md`.
- `work/exp19/` (aggregates only, no per-frame data): `registration.sha256`, `part1-instrument.json`, `part1.json`,
  `part1-rows.json`, `diag-part1.json`, `diag-dead.json`, `part2-pff-rows.json`, `part2-oof-rows.json`, `gate-P.json`,
  `adoption-match-v10-dryrun.json`, `v10-pff-digests.json`, `pff-label-stats.txt`, `tests-before.log`,
  `tests-after.log`, and run logs (`arm_p_train.log` and others).
- Own lines appended to `decisions/log.md`. No frozen file, `HANDOFF.md`, `Mastaadmi.csv`, `docs/`,
  `scripts/export_public.py` or `.gitignore` was edited. A draft `v11/` was written while the folds trained and deleted
  after the gate failed.
- Ignored caches: `external/pff-cache/` (load, views, v10 probabilities), `external/exp19/arm_p/` (models T and
  folds 0-4, held-out q and c).
- Tests: before 99 + 7 pass; after 107 (with the 8 new PFF tests) + 7 pass.

Allowlist lines for `.gitignore` (the parent's file):

```
!/pff/
/pff/*
!/pff/*.py
!/exp19/
/exp19/*
!/exp19/*.py
!/work/EXPERIMENT-19.md
!/work/exp19/
/work/exp19/*
!/work/exp19/*.json
!/work/exp19/registration.sha256
!/work/exp19/pff-label-stats.txt
!/work/exp19/tests-*.log
!/work/exp19/arm_p_train.log
!/outputs/experiment-19-results.md
```

The PFF-derived aggregates should stay out of the public export until Nilay has the PFF licence terms.
