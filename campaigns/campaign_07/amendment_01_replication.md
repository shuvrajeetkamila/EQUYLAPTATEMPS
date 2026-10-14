# Amendment 01 — Seed-123 Replication Registration (2026-10-14)

Filed **after** screen analysis (run `38542143640`) and **before** any
replication training, exactly as the frozen preregistration requires.

Screen outcome (recorded, unmodified): G-B failed (pretrained macro 8.5719 vs
twin 7.5349; pretrained 13.8% worse) and G-C failed (language 1.899× host;
macro > 6.00). The preregistration nonetheless mandates the seed-123
replication of the primary comparison regardless of screen direction, to test
cross-seed stability before final classification.

Registered replication runs (`experiment_registry_replication.json`):
`c07_dense_pre_123`, `c07_dense_rand_123` — identical to the screen pair except
`seed=123`. No design, threshold, sector, or budget changes.
