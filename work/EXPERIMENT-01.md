# External proxy experiment 01, registered before candidate outcomes

Date: 28 October 2026. Purpose: select a first general temporal model without tuning on the competition match.

Sources: https://github.com/metrica-sports/sample-data at e706dd506b360d69d9d123d5b8026e7294b13996. Metrica publishes anonymised sample matches for football analytics and requests attribution for public use. No standalone redistribution licence is supplied; retain data locally and cite the download source, do not relicense or bundle it.

Split: Sample Game 1 development; Sample Game 2 one-time holdout. Game 3 is reserved, not used in this experiment. No competition labels may be read by selection code. All candidate parameter values are fixed below before scores exist.

Target proxy: generate dense H/A/D intervals from event definitions. PASS starts and SET PIECE starts anchor owning team; RECOVERY ends anchor regained ownership; BALL OUT ends, FAULT RECEIVED starts, and SHOT ends with OUT or GOAL anchor dead ball. Carry the latest anchor until the next; exclude leading unanchored frames and reset at each period. Conflicting simultaneous anchors are excluded and reset ownership until a subsequent unambiguous anchor. This is an event-derived proxy, not Metrica corrected possession truth. Unrecorded control changes and event timestamp error limit the result. Labels never enter inference features.

Comparators: official nearest-player baseline and +/-50-frame majority smoothing; temporal H/A/D decoder with team-transition penalties {5,15,40} and equal dead-transition penalty. Spatial distances in metres, per-frame cost scaled by 1/25 s. Fixed distance scale 2 metres; fixed stationary-window criterion 1 second; missing-ball gaps shorter than 3 seconds carry neutral evidence, not dead ball. No labelled tuning of feature formulas during this run.

Conditions: clean sample tracking, and deterministic missing-ball stress (seed 20261028, 41% of frames in contiguous blocks of 50 frames). Stress is synthetic and not claimed to reproduce organiser missingness. Identical masks per method.

Selection: highest development mean spellF1 across the two conditions; ties within 0.005 choose lower transition penalty. Evaluate selected candidate once on Game 2; do not retune on holdout. Report all development arms. Publish holdout proxy spellF1 and macroF1 for selected versus baseline, both conditions, without qualification predictions. A candidate is experimental unless holdout spellF1 exceeds the baseline in both conditions and clean macroF1 is no more than 0.03 lower. This gate is provisional across only one held-out match; it cannot establish competition quality.

Validation: synthetic metric perfect/constant/shifted controls; loader one-based frame check; parser rejection on malformed inputs; synthetic turnover/loose-ball/stoppage and missing-gap cases; deterministic output hashes. Run competition inference locally only after settings freeze. Do not score the new candidate on the competition public window in this experiment.

Route: bridge returned claude/claude-opus-5-5/high/read-only/ponytail for review. Parent owns all implementation and project state. Bridge previously refused the project trust root; attempt normal review once after code exists and report any unresolved review access. No publication or submission.
