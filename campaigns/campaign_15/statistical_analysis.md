# Campaign 15 — Statistical Analysis

Metric: macro dev token NLL (lower better), computed from the
SHA-verified raw records by `publish_campaign15.py`. n = 3 paired
seeds; all statistics descriptive — no significance claims.

## Paired differences (old − new; positive favors the new connector)

| Step | Seed 42 | Seed 123 | Seed 31415 | Mean | Median | SD | 95% paired-t interval (df=2) |
|---|---|---|---|---|---|---|---|
| 0 | +2.233 | +1.247 | +2.290 | +1.923 | +2.233 | 0.587 | [+0.47, +3.38] |
| 8 | −2.383 | −2.295 | −2.267 | −2.315 | −2.295 | 0.060 | — |
| 16 | −1.923 | −1.804 | −2.098 | −1.942 | −1.923 | 0.148 | — |
| 32 | −1.005 | −0.854 | −1.111 | −0.990 | −1.005 | 0.129 | [−1.31, −0.67] |
| 64 | −0.159 | −0.087 | −0.191 | −0.146 | −0.159 | 0.053 | — |
| 96 | +0.251 | +0.113 | +0.109 | +0.158 | +0.113 | 0.081 | [−0.04, +0.36] |

## Reading

- **Step 0 (initialization):** the new connector starts better at
  every seed (Gate B passes; mean +1.92, interval excluding 0).
  The old connector's start varies by seed (13.26–14.31, seam-init
  RNG); the new connector's start is seed-independent (12.0150)
  because zero initialization has no randomness.
- **Step 32 (primary endpoint):** the old connector leads at
  every seed (mean −0.99, SD 0.13, interval fully negative). The
  Campaign 14 single-seed inversion is **not** seed noise — it
  replicates 3/3. Gate C fails.
- **Steps 64→96:** the gap closes monotonically and reverses: at
  step 96 the new connector leads at every seed (mean +0.16),
  though the paired interval includes 0 — direction is consistent
  (3/3) but magnitude is small and cannot exclude a tie at n=3.
- **Absolute scale:** both conditions remain far worse than the
  native recipient (best child 6.78 vs native 2.77): placement
  damage, which no connector can repair, dominates these budgets.
  Winning this comparison is not evidence of a good model.
