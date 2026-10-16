# Campaign 19 — Decision rules (frozen, pre-authorization)

Precedence (first match):
1. `BLOCKED_RECONSTRUCTION` — identity control B fails frozen exactness.
2. `BLOCKED_DONOR_INVALID` — donor-validity gate D0 fails.
3. `INCONCLUSIVE_CONTROL_FAILURE` — D or E fails to train/evaluate, or F/G behaves inconsistently (ablation does not change C, or restoration does not reproduce C).
4. `TRANSFER_EVIDENCE_POSITIVE` — all primary success criteria met (B exact; ≥0.25 mean advantage vs best control at step 32; ≥2/3 seeds positive; ablation removes ≥50%; preservation ≤ +4.50).
5. `DONOR_ADVANTAGE_NOT_SHOWN` — controls valid, criteria not met.

`TRANSFER_EVIDENCE_POSITIVE` would mean donor-specific
learned-structure advantage on the recipient panel — not proof of
general capability transfer, which additionally requires the sealed
held-out stage and a capability the donor demonstrably owns.
