# Campaign 16 — Frozen Decision Rules

Metric: macro dev token NLL (lower better).
`delta_t(lr, s) = old_macro − new_macro`; positive favors new.

**Ranking(lr, step):** NEW if mean delta ≥ +0.25 with ≥2/3 seeds
positive; OLD if mean delta ≤ −0.25 with ≥2/3 seeds negative;
else TIE.

**Gate E signature (all required):** (i) at baseline LR, the
median steps 1–32 pre-clip gradient-norm ratio new/old is on the
same side of 1 at all three seeds with the median ratio outside
[0.8, 1.25]; (ii) the LR response matches the gradient
prediction: if new's gradients are smaller (ratio < 1),
mean delta_32 at 3× LR ≥ mean delta_32 at 0.3× LR + 0.25 (and
symmetrically if ratio > 1).

**Primary outcome — apply in precedence order:**
1. `IMPLEMENTATION_FAILURE` — Gate A fails or Gate B integrity
   fails for any primary run.
2. `INCONCLUSIVE` — fewer than 18 valid primary runs.
3. `NO_RELIABLE_DIFFERENCE` — ranking is TIE at steps 32 and 96
   at every LR level.
4. If step-32 rankings across LR levels contain both NEW and OLD:
   `OPTIMIZATION_MECHANISM_SUPPORTED` if the Gate E signature
   holds, else `LEARNING_RATE_SENSITIVE`.
5. If step-32 rankings are not all equal across LR levels (LR
   modulates strength without full crossover):
   `LEARNING_RATE_SENSITIVE`.
6. If the ranking flips between steps 32 and 96 at ≥2 LR levels:
   `TRAINING_BUDGET_DEPENDENT`.
7. Otherwise: `CONNECTOR_DIFFERENCE_PERSISTS`.

Secondary statements (always reported): Gate E supported or not
(regardless of primary label), the placement-damage context
(native 2.7731; placement-only 12.0150), and the explicit caveat
that LR sensitivity, if found, is *consistent with* — not proof
of — learning rate as the sole cause (Part 10.3 of the brief).
