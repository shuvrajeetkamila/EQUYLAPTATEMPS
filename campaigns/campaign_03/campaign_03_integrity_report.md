# Campaign 3 — Integrity Report

- Frozen-parent fingerprints were verified unchanged at the end of all three
  connected runs (`parents_unchanged: true` in every session summary).
- Every candidate's bridge-only reference was reconstructed in-job and
  reproduced identical logits (`reference_reconstruction_identical_logits:
  true` for all non-random candidates; random-sector controls cannot be
  exported as pretrained children by design).
- Evidence bundles were imported from PR comments with SHA-256 verification
  (runs `38367618192`, `38371104410`, `38375777144`) and committed unchanged
  under `campaigns/campaign_03/screen_*`, `confirm_*`, `confirm2_*`.
- Candidate binaries/optimizer states remain in the identified GitHub
  Actions artifacts (`campaign03-screen-recovery`, `campaign03-confirm-recovery`,
  `campaign03-confirm2-recovery`); they were not downloaded in this sandbox,
  so local candidate-binary reverification is incomplete — recorded, not hidden.
- `tests/test_campaign03_results.py` recomputes every headline number from
  the raw records; checksums detect inconsistent records but cannot
  authenticate a maliciously forged complete bundle.
- Tests at publication: see final report. Standard CI remains offline;
  real-parent jobs are separately opted in by commit-message markers.
