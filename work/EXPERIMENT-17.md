# Experiment 17: video ball detection to fill missing-ball frames (v9)

Registered 6 November 2026, before any detector output on any footage. v8 stays the submission unless
every check below passes. Nilay chose the detector and accepted the ISSIA terms (decisions/log.md, 6 Nov).

## Idea

After broadcast fusion (v8), 37.6% of graded tactical frames still have no ball, and the two cameras lose
the ball together. The organisers allow the match video as an algorithm input provided nothing is trained,
fitted or calibrated on it (clarification of 4 Nov). An externally trained ball detector run on the
tactical video may find the ball in some of those frames. EXPERIMENT-13/16 showed that recovering missing
ball is the largest lever we have (+0.018 to +0.043 on the analogue for 14 to 20 points of recovery).

## Detector (fixed)

- WASB (Tarashima et al., BMVC 2023), MIT code, commit 923462c of github.com/nttcom/WASB-SBDT, HRNet with
  3 input and 3 output frames, soccer weights `wasb_soccer_best.pth.tar` SHA-256
  d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c, trained by its authors on ISSIA-CNR
  cameras 1 to 4 (ISSIA terms: non-commercial research in video signal processing). Loaded with
  `torch.load(weights_only=True)`; never retrained or fine-tuned.
- Input: non-overlapping windows of 3 consecutive frames (WASB step 3), ImageNet normalisation, 512x288.
- Output per frame: sigmoid heatmap; connected components above 0.5 (WASB default); each blob gives a
  heat-weighted centroid and its peak value. The frame's detection is the blob with the highest peak.
- Scale rule (from geometry only): WASB sees ISSIA frames shrunk 3.75 times; ISSIA shows about 53 px per
  metre (ball about 12 px), so the model sees about 14 px per metre. The tactical homography gives a median
  of 29 px per metre at the pitch centre (22 to 44 across the pitch; computed from the homography CSV, no
  video, no labels). Tactical tiles of 1024x576 are therefore resized by 0.5 to 512x288. Grid: x0 in {0, 896},
  y0 in {0, 504}; per frame the highest-peak blob over the 4 tiles is kept, in full-frame pixels.

## External screen (ISSIA held-out cameras 5 and 6, `v9/issia_screen.py`)

Ground truth: WASB's corrected ISSIA labels, in-game ball only. Relative scale k in {0.75, 1.0, 1.5}
(model-input ball size relative to WASB's native setting; k = 1.0 is native), applied by the same tiler.
A detection is correct if within 15 original ISSIA pixels (about 0.28 m) of the label.

- recall = correct detections / labelled frames (visible and occluded);
- precision = correct detections / all detections (a detection on a frame without a label is false).

Threshold rule: tau = the smallest peak threshold in {0.5, 0.6, 0.7, 0.8, 0.9} with precision >= 0.95 at
k = 1.0 (cameras 5 and 6 pooled). **Stop** if no tau qualifies, or if recall at tau is below 0.60 at k = 1.0
or below 0.40 at k = 0.75.

## Fusion (fixed, `v9/infer.py`)

Start from the v8 fused ball. A frame still missing the ball takes the video detection if its peak >= tau,
its pixel maps through the inverse of that frame's tactical homography into normalised pitch coordinates,
and that point lies within 2 m outside the pitch at most. Then the frozen v6 model, features and decoder,
exactly as v8. The detections are an intermediate file bound by SHA-256.

## Instrument checks on the competition match (label-free, Mac only, aggregates only, one shot)

All must hold or the experiment stops and v8 stays:

- I1: on frames where the ATD ball is visible and the detection passes tau, the median distance between
  them is <= 1.0 m over at least 10,000 frames;
- I2: on those frames, at most 10% are more than 3 m apart;
- I3: the missing-ball share of graded frames after the v9 fill is <= 0.276 (at least 10 points below
  v8's 0.376).

No threshold, scale, tile, rule or model is changed after seeing these numbers.

## Adoption (if the screen and I1 to I3 pass)

Freeze v9 sources and the detections digest; two complete runs (detector and inference) with identical
CSV digests (if the detector's MPS output differs between runs, both runs are repeated on CPU); official
validator; Smart Tagging phase agreement >= v8 - 0.02 (v8 0.783). Then copy to Mastaadmi.csv.

Reading of I3 (diagnostic, from EXPERIMENT-16, not a gate): 10 points of recovery suggests roughly +0.01
to +0.015 spellF1; 20 points roughly +0.03 to +0.04, if the recovered frames are as useful as random ones.

## Limits

- No external footage with possession labels exists for a downstream test; the screen measures the
  detector only, and the gain is inferred from the EXPERIMENT-13/16 analogue.
- ISSIA cameras are static; the tactical camera pans, which may hurt a 3-frame detector.
- Aerial balls are projected onto the ground plane and misplaced.
- Recovered frames may be easier than the average missing frame (the analogue recovers at random).
- ISSIA's terms restrict use to non-commercial research; weights and data are fetched by script and never
  committed, and the terms are disclosed in THIRD_PARTY.md.

## Implementation notes (added 6 November, before any detector output on ISSIA or the match)

- The model runs in half precision on the Mac GPU (MPS): 55 tiles/s against 21 in full precision; on 32
  ISSIA tiles the largest heatmap difference was 0.002 and the same blobs were found. Used for the screen
  and the match alike. Same-device repeat runs were identical on 48 frames.
- Tiling follows `v9/wasb.py` (`tiles`, `merge`); the tail of a video is covered by one extra window ending at
  the last frame.
- Independent review (Codex, 6 Nov, before any match output) found two defects, both fixed before the match
  run: I1/I2 now use every detection passing tau, on or off the pitch, as registered (the first version
  dropped off-pitch detections before measuring agreement); pitch length is 105.3 m as in `games.py` and v8
  (the ATD metadata says 105).
