# Campaign 17 — Frozen Decision Rules

Apply in precedence order:

1. `PLACEMENT_IMPLEMENTATION_FAILURE` — Gate A or Gate B fails,
   or any manifest contradicts measured fingerprints (Gate C).
2. `INCONCLUSIVE` — required records missing/invalid such that
   gates cannot be applied.
3. `READY_FOR_CROSS_FAMILY_PILOT` — Gates A–C pass **and** the
   D1 long-range exchange degradation ≤ 0.5 NLL (matched-depth
   relocation is gentle even across distant positions).
4. `INTERFACE_REPAIRED_STRUCTURAL_DAMAGE_REMAINS` — Gates A–C
   pass but D1 degradation > 1.0 NLL (the interface is correct;
   block position itself carries function).
5. `NATIVE_DEPTH_VALIDATED` — Gates A–C pass and D1 degradation
   in (0.5, 1.0] (placement preserved; exchange cost modest but
   above the pilot-readiness bar).

Gate F readiness follows the label: only
READY_FOR_CROSS_FAMILY_PILOT authorizes designing a donor pilot;
the other passing labels route to placement research first.
