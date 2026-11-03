# Third-party provenance

`starter-repo/` contains selected public source and documentation vendored without changes from https://github.com/metrica-sports/more-than-a-hack-2027 at commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`. The original MIT notice is retained in `starter-repo/LICENSE` and applies to those upstream files. Selection: root Markdown/CFF/licence/requirements plus Python modules under starter/ and evaluation/. Public ground-truth CSVs, baseline prediction CSVs, notebooks, data and upstream Git metadata are intentionally omitted from this cloud package. Retrieve public documentation from the upstream link if needed; do not fetch the five-minute labels for model improvement.

`work/learned-evaluation/model.pkl` was trained locally using public Metrica Sports Sample Game 1 event/tracking data. Public external dataset source: https://github.com/metrica-sports/sample-data at commit `e706dd506b360d69d9d123d5b8026e7294b13996`. Authors request attribution. No standalone redistribution licence was supplied, so raw datasets and event-definition extracts are not bundled or relicensed. External data hashes and training split are recorded in work/hybrid-evaluation/provenance.json and the experiment registrations.

Nilay's original work is licensed MIT (root `LICENSE`, chosen 3 November 2026). Competition match assets are separately restricted, absent from this repository and never covered by the organiser's code licence.

`v2/` reads the IDSSE/DFL open dataset: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195; figshare doi 10.6084/m9.figshare.28196177.v1, licence CC BY 4.0. Raw files are fetched into ignored `external/idsse/` only (see HANDOFF.md) and are not bundled. Any derived public material must carry this attribution.

`v3/` evaluation reads SkillCorner open data (https://github.com/SkillCorner/opendata, commit 4340d274572876239c154c90bc507a9b3250a656), MIT licence, Copyright (c) 2020 SkillCorner. Raw files stay in ignored `external/`; SkillCorner asks users to credit them.
