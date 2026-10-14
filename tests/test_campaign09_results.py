"""Campaign 9 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns"
JOBS = {
    "c09_natgeom_pre_42": "campaign_09/screen_38578925047",
    "c09_natgeom_rand_42": "campaign_09/screen_38578925047",
    "c09_natgeom_pre_123": "campaign_09/replication_38583133395",
    "c09_natgeom_rand_42_96": "campaign_09/budget96_38583138287",
    "c07_dense_pre_42": "campaign_07/screen_38542143640",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign09_outcome_b_traces_to_records():
    results = json.loads((ROOT / "campaign_09" / "results.json").read_text())
    assert results["outcome"]["classification"] == "B"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    for exp_id, folder in JOBS.items():
        record = json.loads((ROOT / folder / "runs" / exp_id / "results.json").read_text())
        assert results["runs"][exp_id]["macro_dev_nll"] == macro(record["validation"])
    natgeom = [c for c in results["comparisons"] if c["condition"] == "natgeom"]
    assert len(natgeom) == 3
    # Donor-specific advantage replicated, usefulness never achieved.
    assert all(c["G_B_pass"] and not c["G_C_pass"] for c in natgeom)
    assert all(c["benefit_fraction"] > 0.09 for c in natgeom)
