# Campaign 3 — Initial Audit (Stage 1)

Date: 2026-10-13. Project: **HETEROGENEOUS AI MODEL SURGERY** (existing
repository `shuvrajeetkamila/EQUYLAPTATEMPS`; internal package/CLI names are
unchanged compatibility identifiers).

## Verified starting state

| Item | Reported | Verified actual | Status |
|---|---|---|---|
| Branch | `arena/97efdf4a-equylaptatemps` | same, checked out | PASS |
| HEAD | `5ac671aa9963b90577393f40740d397bdc047341` | identical local and remote head | PASS |
| Working tree | — | clean before Campaign 3 work | PASS |
| PR | #1 open | OPEN, head `5ac671a`, base `main` | PASS |
| Offline CI | Python 3.10/3.11 pass | run `38363810539`: both jobs pass | PASS |
| Campaign 2 outcome | Outcome C | `campaigns/campaign_02/campaign_02_results.json` confirms C | PASS |
| Local tests at audit | 211 reported | 211 reproduced pre-Campaign-3; 221 after new seam tests | PASS |

## Parents (immutable, verified in Campaign 2 records)

- Recipient/host: `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
  (selective SSM + causal convolution, width 768, 24 blocks, 129,135,360 params).
- Donor: `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`
  (GQA/RoPE transformer, width 1536, 28 blocks, 1,543,714,304 params).

## Recovered vs unavailable artifacts

Recovered/verified locally: both published bridge-only interface checkpoints
(pilot SHA-256 `5f233721…`, structural `bcc43ee4…`, byte-unchanged), the
synthetic baseline ZIP, all Campaign 2 raw records, registries, journals, and
reports. The Campaign 2 candidate binaries remain only in GitHub Actions
artifact `campaign02-alignment-recovery` (run `38358962201`, 200,690,671 bytes,
not expired, re-verified via API in this audit); they are **not downloaded
locally** and are not needed: Campaign 3 retrains its own dense-covariance
control under its registered data protocol rather than trusting binaries it
cannot inspect. Local reverification of those binaries remains incomplete,
exactly as Campaign 2 recorded.

## Environment

Local sandbox: 2 CPUs, 3.8 GB RAM, no GPU, Hugging Face unreachable (TLS EOF).
Real-parent experiments therefore run only in the separately opted-in GitHub
Actions CPU workflows, as in Campaigns 1–2. Standard CI stays offline.

## Best Campaign 2 bridge (starting point)

`primary_covariance_42`: LayerNorm→full dense seams with covariance-whitening
initialization + scalar RMS gains (input gain 36.3989, output gain 1.2922),
32 AdamW steps, seed 42. Development macro NLL 5.6783 (language 6.7650,
math 4.5917) vs legacy random bridge 7.5882. It failed the frozen promotion
gate: language ratio 1.539× host vs 1.50 maximum. That bridge is retained
unchanged as the Campaign 3 dense control (`c03_dense_cov_42`).
