# Campaign 18 — Damage predictor specification

Frozen in `damage_predictor.json` before the held-out set was
opened (opening recorded in `heldout_access_log.json`).

- Candidates (preregistered): P0 constant development-mean damage;
  P1 OLS on max |displacement|; P2 OLS on max |displacement| +
  number of displaced blocks + mean normalized source position.
- Fit: development (48) only. Selection: validation (5) MAE,
  ties toward simpler. Chosen: **P2_position_aware**
  (validation MAE 0.969 vs P1 1.006, P0 2.570).
- Held-out confirmation (single opening, run 38911632615):
  P2 MAE **0.852** vs P0 1.648 and P1 0.874 — ratio 0.52 ≤ 0.8
  frozen threshold ⇒ **Gate G3 PASS**.
- Limits: coefficients in `damage_model.json`; worst held-out
  misses listed there. The predictor ranks/estimates structural
  damage for *recipient self-moves*; it says nothing about foreign
  weights and was never refit against held-out outcomes.
