# EXPERIMENT-07 result: label-free Smart Tagging corroboration (3 November 2026)

Run on Nilay's Mac with `v4.smart_tagging_audit`, against frozen v2 and v4 on the competition tracking. Only aggregates were returned; no labels were read.

The registered measure is pooled agreement with the possession team over BUILD UP, PROGRESSION, ATTACKING TRANSITION and LONG BALL tag frames.

| Model | Agreement |
|---|---:|
| v2 | 0.6874 |
| **v4** | **0.7915** |

**Verdict: corroborated.** v4 exceeds v2 by 0.104; the rule only required no worse than −0.02.

## Descriptive (not gated)

- **Agreement by code, v2 → v4:**
  - BUILD UP 0.777 → 0.914; PROGRESSION 0.722 → 0.844; LONG BALL 0.548 → 0.631; ATTACKING TRANSITION 0.608 → 0.634.
  - Mirrored defending codes, measured as agreement with the *defending* team: v2 0.14–0.20, v4 0.02–0.08.
- **Dead share inside open-play phases** (BUILD UP, PROGRESSION, transitions): v4 0.07–0.09, similar to v2.
- **Dead share inside chance and box phases:**
  - CREATING CHANCES: v4 0.50, v2 0.41.
  - PLAYERS IN THE BOX: v4 0.48, v2 0.42.
  - BALL IN THE BOX: v4 0.36, v2 0.30.

  Possible over-calling of dead ball while the ball is missing in crowded boxes. Unresolved without labels; it is a candidate for future work.
- **Restarts:** both models show a dead-to-live switch near 54 of 59 tagged set pieces. Offset p10/p50/p90: v2 −0.1 / 6.1 / 13.0 s, v4 2.4 / 6.2 / 10.3 s.

Smart Tagging is automatic and noisy. Agreement is a proxy for, not a measure of, competition accuracy.
