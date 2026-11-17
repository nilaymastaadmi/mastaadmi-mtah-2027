# EXPERIMENT-27 results: PASS

**Verdict: PASS.** Registered bar (`work/EXPERIMENT-27.md`, sha256 9a6cad62, hash re-checked by the gate) applied mechanically: T2 meets checks 1 to 7 (`work/exp27/gate.json`), and the built candidate passes every real-match guard of rule v2 clause (4) (`work/exp27-evaluation/freeze.json`). Candidate CSV sha256 045ed8a5 in `work/exp27-evaluation/`. `Mastaadmi.csv`, `v12/` and the public repo are untouched; adoption is the parent's call.

Candidates: v12's frozen models (conv recipe M seed 22 and v8's boosted model) applied to four symmetric views of the same tracking (identity; touchline mirror; full swap with teams swapped; goal-line mirror with teams swapped), features recomputed per view, H and A mapped back, probabilities averaged. T1 averages the conv half only, T2 both halves; v12's decoder. No model was trained or tuned.

## Bar (rule v3 for EXP-27, copied unchanged; all must hold)

| check | needed | T1 | T2 |
|---|---|---|---|
| PFF-64 mean-of-4 delta | >= +0.003 | **+0.0009** (fail) | +0.0035 |
| PFF-64 wins | >= 36 of 64 | 37 of 64 | 43 of 64 |
| PFF-64 controlled3 delta | >= 0 | +0.0012 | +0.0041 |
| PFF-64 raw delta | >= 0 | +0.0007 | +0.0032 |
| SkillCorner OOF mean-of-4 delta | >= 0 | +0.0017 | +0.0009 |
| DFL gaps OOF mean-of-4 delta | >= -0.005 | -0.0008 | +0.0032 |
| DFL gaps OOF raw delta | >= -0.005 | -0.0001 | +0.0049 |
| checks 1 to 7 | | fail | pass |

## Real-match guards (rule v2 clause 4) on the built candidate

| guard | value | needed | result |
|---|---|---|---|
| two byte-identical full runs | 045ed8a5 / 045ed8a5 | equal | pass |
| official validator (both runs) | passed | passed | pass |
| possession starts inside quiet stretches | 22 | < 27 | pass |
| Smart Tagging phase agreement | 0.7879 | >= 0.763 | pass |
| predicted D share | 0.4854 | 0.283 to 0.542 | pass |
| median possession length (s) | 8.84 | 3.0 to 10.12 | pass |

Build: v12 tracking path, 4 views, frozen v6 + final M (e68c5767), mean over views, 50/50, v12 decoder; warm-up fix (frames 110,808 to 113,741 get period 0 for features, conv context and decoding) and pre-kickoff fix (those frames set to D); two full runs from scratch.

Against the current `Mastaadmi.csv` (sha256 0cdc7351) the candidate changes 1,909 of 199,319 frames (984 before the second half's graded block, 0 in the pre-kickoff window, 925 after the kickoff). T2 with and without the warm-up fix differ on 153 frames, all after the kickoff (`work/exp27-evaluation/diff-vs-submission.json`).

Limits: the PFF-64 margin over the +0.003 bar is small (bootstrap interval below); DFL clean, outside the bar, is -0.0007 mean-of-4; the warm-up fix is in the build but not in any out-of-fold or PFF number (external matches have no pre-kickoff gap); PFF-64 has judged at least 11 earlier candidates.

## All arms against v12 (mean-of-4 delta (wins) / raw delta)

| arm | PFF-64 | SkillCorner OOF | DFL gaps OOF | DFL clean OOF | Metrica clean (diagnostic) | Metrica gaps (diagnostic) |
|---|---|---|---|---|---|---|
| T1 (conv averaged) | +0.0009 (37/64) / +0.0007 | +0.0017 (14/20) / +0.0014 | -0.0008 (4/7) / -0.0001 | +0.0019 (4/7) / +0.0030 | +0.0031 (2/3) / +0.0007 | +0.0113 (3/3) / +0.0101 |
| T2 (both averaged) | +0.0035 (43/64) / +0.0032 | +0.0009 (12/20) / +0.0012 | +0.0032 (4/7) / +0.0049 | -0.0007 (4/7) / +0.0016 | +0.0050 (2/3) / +0.0016 | +0.0072 (3/3) / +0.0049 |
| V1 alone (touchline mirror) | +0.0000 (0/64) / +0.0000 | +0.0000 (0/20) / +0.0000 | +0.0000 (0/7) / +0.0000 | +0.0000 (0/7) / +0.0000 | +0.0000 (0/3) / +0.0000 | +0.0000 (0/3) / +0.0000 |
| V2 alone (full swap) | -0.0002 (34/64) / +0.0003 | +0.0011 (10/20) / +0.0017 | -0.0004 (4/7) / +0.0038 | -0.0037 (3/7) / +0.0003 | +0.0017 (2/3) / -0.0007 | -0.0009 (1/3) / -0.0032 |
| V3 alone (goal-line mirror, teams swapped) | -0.0002 (34/64) / +0.0003 | +0.0011 (10/20) / +0.0017 | -0.0004 (4/7) / +0.0038 | -0.0039 (3/7) / +0.0001 | +0.0017 (2/3) / -0.0007 | -0.0009 (1/3) / -0.0032 |

T2 on PFF-64: mean-of-4 bootstrap 95% [+0.0013, +0.0057], by convention raw +0.0032, controlled1 +0.0033, controlled2 +0.0035, controlled3 +0.0041; frames changed against v12, median 1,794 per match.

## The four views collapse to two

Every input column is symmetric under the touchline mirror up to float rounding (largest column difference on three checked views 2e-6), so V1 decodes like V0 (at most 4 frames differ on any of the 104 views) and V3 like V2 (97 of 104 views give identical rows). The registered four-view average therefore acts as an equal-weight average of the identity and the full team swap. This was not known at registration; the candidate scored is exactly the registered one.

## Predictions (registered before any score)

| | prediction | outcome | |
|---|---|---|---|
| P1 | median OOF view: T1 changes < 2% of graded frames against V0 | 0.74% | right |
| P2 | T1 changes PFF-64 mean-of-4 by -0.002 to +0.004 (check 1 likely fails) | +0.0009, check 1 fails | right |
| P3 | T2 above T1 on PFF-64 mean-of-4 | +0.0035 against +0.0009 | right |
| P4 | V1 is the view closest to V0 | V1 changes at most 4 frames | right |

## Label-free check on the competition match (no warm-up fix; pre-kickoff fix applied to every arm)

Share of graded frames whose argmax differs between any two views: conv 0.0539, boosted 0.0376, blend 0.0320; blend against V0 per view: V1 0.0000, V2 0.0320, V3 0.0320. The V0 pipeline reproduces `Mastaadmi.csv` frame for frame: yes.

| arm | possession starts | per minute | starts in quiet stretches | D share | median possession (s) |
|---|---|---|---|---|---|
| v8 (bb0ba6f5) | 253 | 2.389 | 18 | 0.4751 | 8.36 |
| v12 | 222 | 2.096 | 21 | 0.4850 | 8.88 |
| T1 | 225 | 2.125 | 22 | 0.4840 | 8.88 |
| T2 | 224 | 2.115 | 22 | 0.4849 | 8.84 |

## Process

- H1: V0 recomputed through this pipeline equals v12's stored conv probabilities on 40 of 40 OOF views and 64 of 64 PFF matches (byte-identical) and the boosted probabilities on 40 and 64; V0-only scoring reproduces EXPERIMENT-22's v12 rows with max difference 0.0 and 0.0.
- H3: the full swap rebuilt from tracking equals EXPERIMENT-22's cached swap view exactly: yes. On the match, V0's boosted half reproduces v8's frozen label digest.
- PFF-64 opened once: `2026-11-17 02:49:17 IST open reason: EXPERIMENT-27 sealed PFF-64 stage: harness passed and oof.json exists; scored once as registered`.
- Out of fold, the boosted half used v8's stored v6-recipe fold models (`external/models/`), so T2 is out of fold on SkillCorner and DFL; Metrica and PFF used the frozen v6 model, as v12 does.

## Files

Registration `work/EXPERIMENT-27.md` (sha256 in `work/exp27/registration.sha256`); code `exp27/`; views and predictions `external/exp27/`; scores `work/exp27/{harness,oof,oof-rows,pff,pff-rows,gate,match-check}.json`; build and freeze `work/exp27-evaluation/`.
