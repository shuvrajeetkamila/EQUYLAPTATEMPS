"""Campaign 10 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_10"
JOBS = {
    "c10_out_pre_42": "screen_38602501026",
    "c10_out_rand_42": "screen_38602501026",
    "c10_out_awake_42_amd1": "awakeamd_38606201339",
    "c10_out_pre_123": "replication_38606208077",
    "c10_out_awake_42_96": "budget96_38607522188",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign10_outcome_b_traces_to_records():
    results = json.loads((ROOT / "results.json").read_text())
    assert results["outcome"]["classification"] == "B"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    for exp_id, folder in JOBS.items():
        record = json.loads((ROOT / folder / "runs" / exp_id / "results.json").read_text())
        assert results["runs"][exp_id]["macro_dev_nll"] == macro(record["validation"])
    primary = [c for c in results["comparisons"] if c["steps"] == 32]
    assert all(c["G_B_vs_D_pass"] and not c["G_C_pass"] for c in primary)
    budget = next(c for c in results["comparisons"] if c["steps"] == 96)
    assert budget["benefit_vs_C_fraction"] < 0  # fragility at 96 steps is part of the record
