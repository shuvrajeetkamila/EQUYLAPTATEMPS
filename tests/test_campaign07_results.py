"""Campaign 7 published results must match the imported run records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_07"
JOBS = {
    "c07_dense_pre_42": "screen_38542143640",
    "c07_dense_rand_42": "screen_38542143640",
    "c07_dense_pre_123": "replication_38544932532",
    "c07_dense_rand_123": "replication_38544932532",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign07_outcome_traces_to_records():
    results = json.loads((ROOT / "campaign_07_results.json").read_text())
    assert results["outcome"]["classification"] == "C"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    for exp_id, folder in JOBS.items():
        record = json.loads((ROOT / folder / "runs" / exp_id / "results.json").read_text())
        assert results["runs"][exp_id]["macro_dev_nll"] == macro(record["validation"])
    for seed in ("42", "123"):
        gate = results["gates"][seed]
        assert gate["G_B_pass"] is False and gate["G_C_pass"] is False
        assert gate["benefit_fraction"] < 0  # twin beat pretrained at both seeds
