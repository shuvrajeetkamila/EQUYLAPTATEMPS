# Campaign 7 — Cross-Pairing Replication

**Pairing (frozen):** Mamba-130M host × SmolLM-135M (Llama) donor — the first
donor-family change in the project. **Status:** COMPLETE — Outcome C (see final report).

| File | Contents |
|---|---|
| `prior_campaign_evidence_audit.md` | What Campaigns 1–6 did/did not establish |
| `pairing_selection.md` | Candidate table, pinned revisions, rejected alternatives |
| `preregistration.md` / `.json` | Frozen design, gates, outcome mapping |
| `experiment_registry.json` | Screen runs (pretrained vs matched random twin) |
| `experiment_matrix.csv` | Run matrix and statuses |
| `compatibility_report.md` | Stage-1 compatibility evidence (after screen job) |
| `control_comparisons.csv`, `donor_ablation_results.csv` | Twin and ablation tables (after analysis) |
| `integrity_report.md`, `resource_report.md`, `failure_analysis.md` | Audits (after execution) |
| `screening_analysis.md` | Gate evaluation from raw records |
| `test_report.md`, `recovery_guide.md` | Test evidence and artifact recovery |
| `../../HETEROGENEOUS_AI_MODEL_SURGERY_CAMPAIGN_07.md` | Final report (Outcome A/B/C/D) |

Reproduce the screen: trigger `.github/workflows/campaign07-screen.yml` via a
commit containing `[campaign07-screen]`, or run
`python scripts/campaign07_runner.py --registry campaigns/campaign_07/experiment_registry.json --out artifacts/campaign07-screen`
on a machine with ≥12 GiB RAM and network access to Hugging Face.
