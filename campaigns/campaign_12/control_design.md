# Campaign 12 — Control Design

## Eligibility policy (B and C)

Eligible tensors are exactly the donor sector's 2D weight matrices
(7 per block × 6 blocks = 42): per Llama block `q_proj`, `k_proj`,
`v_proj` [576,576]/[192,576] and `o_proj` [576,576], `gate_proj`
and `up_proj` [1536,576], `down_proj` [576,1536]. Excluded: the 12
1D RMSNorm gain vectors (preserved intact; scrambling a per-channel
gain vector would change the normalization semantics rather than
rearrange a learned map, and Campaign 11 set this precedent).
Embeddings are not part of the transplanted sector. No biases exist
in these projections. Shapes and dtypes are preserved by
construction and asserted in tests and run records.

## Condition B — within-row permutation (new in Campaign 12)

For each eligible matrix `W[out, in]`, each output row is permuted
independently (single `torch.Generator` seeded 20241014, rows
processed in recorded tensor order). Preserved exactly, per row:
value multiset, mean, variance, L1/L2 norms; hence also the whole
tensor's multiset and Frobenius norm. Destroyed: which input
coordinate each weight in the row multiplies — i.e. the learned
alignment between input features and each output unit, while each
output unit keeps its own weight-value profile. This is a *finer*
scrambling than condition C in one sense (row statistics survive)
and incomparable in another (C also moves values across rows,
destroying per-unit profiles). The campaign makes no prior claim
that B is "weaker" than C; the ladder is empirical.

Verification (in the control itself, recorded per tensor): sorted
values of each row before vs after must be exactly equal
(`torch.equal` on sorted rows, float32, no tolerance — permutation
only copies values, so exactness is required, not approximate);
row count preserved; tensor SHA-256 before/after recorded;
whole-tensor multiset check retained. Donor parent is deep-copied
before any modification and verified untouched by fingerprint in
tests and by `parent_weights_unchanged` in run records.

## Condition C — full-tensor permutation (Campaign 11 reproduction)

`shuffled_sector_control` reused without modification (same seed
20241014, same eligibility, same per-tensor sorted-multiset
verification and fingerprints). Cross-check: Campaign 12's C
records must show the same eligible count (42), the same excluded
count (12), and `all_multisets_preserved` true, matching Campaign
11's records; the permutation fingerprints will differ from
Campaign 11's only if code changed — they are compared during
publication and any difference reported, not hidden.

## Condition D — random twin (reference)

Established `random_qwen_sector_control` with
`random_sector_seed 1234`, identical interface treatment. Its
known tendency toward near-zero graft update under this interface
(Campaigns 9–10: ≈0.001) is reported as measured; D is a reference
for "any injected sector," not a validity-gated control.

## Activity comparability

Pre-training validation (separate `[campaign12-validate]` job,
before confirmatory training) constructs A/B/C/D children and
records: control records, initial graft activity
(`initial_statistics.json` summary incl. relative update at
initialization), shapes/dtypes, parameter counts, and parent
fingerprints before/after construction. Post-training, each run's
trained graft relative update and bypass/zero ablations are
checked against the frozen validity criteria in the
preregistration. No calibration is applied to force activity
matching: differences are measured and reported, per protocol.
