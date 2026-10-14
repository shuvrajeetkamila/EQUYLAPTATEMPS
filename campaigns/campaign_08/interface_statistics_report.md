# Campaign 8 Interface Statistics — Baseline Report (Stage B)

Source: Campaign 7 stored per-run statistics (job `38542143640`, measured
pre-training on 8 unlabeled TRAIN windows; machine-readable form in
`interface_statistics_baseline.json`).

| Stream | Global RMS | Meaning |
|---|---|---|
| `donor_native_input` | **196.50** | What SmolLM blocks [24,30) receive inside their native model |
| `child_donor_input` | 1.056 | What the Campaign 7 child actually fed them (both children) |
| `recipient_native_output` | 1.434 | Host-stream scale the suffix natively expects |
| `child_host_suffix_input` | 0.922 / 0.952 | What the child fed the host suffix (pre / twin) |
| graft input → output RMS (pretrained, trained) | 1.064 → **57.77** | Sector output magnitude after seam training |
| graft input → output RMS (twin, trained) | 1.065 → 1.187 | Random sector stays near-identity in scale |

## Interpretation (diagnostics only — not causal evidence)

1. There is a large **scale and geometry mismatch** at the donor boundary:
   natively, the sector's input stream has RMS ≈196; the child feeds ≈1.06
   (≈186× smaller), with host-covariance structure rather than donor-native
   structure. Per-token RMSNorm inside each Llama block cancels pure scale,
   so the operative issue is directional/covariance geometry — precisely what
   the frozen `covariance_orthogonal` calibration (whitening + orthogonal
   variance-preserving pair + RMS matching) is designed to address.
2. After training, the pretrained sector inflates its output RMS ~54× while
   the twin stays near 1.1× — the seams learn to *shout* into the sector.
   Whether that shouting carries useful signal is exactly what the matched
   Stage D comparison tests; Campaign 7 says it did not help under `random`
   initialization.
3. These statistics justify *testing* calibration; they do not predict its
   success. H0 remains the default.
