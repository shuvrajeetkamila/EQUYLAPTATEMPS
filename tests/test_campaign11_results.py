"""Campaign 11 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT11 = Path(__file__).parent.parent / "campaigns" / "campaign_11"
ROOT10 = Path(__file__).parent.parent / "campaigns" / "campaign_10"
JOBS = {
    "c10_out_pre_42": (ROOT10, "screen_38602501026"),
    "c10_out_pre_123": (ROOT10, "replication_38606208077"),
    "c10_out_pre_42_96": (ROOT10, "budget96_38607522188"),
    "c11_shuf_42": (ROOT11, "screen_38713676113"),
    "c11_shuf_123": (ROOT11, "screen_38713676113"),
    "c11_shuf_42_96": (ROOT11, "budget96_38715108842"),
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign11_outcome_c_traces_to_records():
    results = json.loads((ROOT11 / "results.json").read_text())
    assert results["outcome"]["classification"] == "C"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    assert results["control_valid"] is True
    assert results["primary_gate_pass"] is False
    for exp_id, (root, folder) in JOBS.items():
        record = json.loads((root / folder / "runs" / exp_id / "results.json").read_text())
        assert results["runs"][exp_id]["macro_dev_nll"] == macro(record["validation"])
        assert record["training"]["parent_weights_unchanged"] is True
    primary = [c for c in results["comparisons"] if c["steps"] == 32]
    assert primary[0]["benefit_percent"] >= 2.0  # seed 42 passes its gate alone
    assert primary[1]["benefit_percent"] < 1.0  # seed 123 fails: no replication
    budget = next(c for c in results["comparisons"] if c["steps"] == 96)
    assert budget["benefit_percent"] < 0  # shuffled control ahead at 96 steps


def test_campaign11_control_validation_traces_to_records():
    validation = json.loads((ROOT11 / "control_validation.json").read_text())
    assert validation["control_valid"] is True
    for exp_id in ("c11_shuf_42", "c11_shuf_123", "c11_shuf_42_96"):
        root, folder = JOBS[exp_id]
        init = json.loads((root / folder / "runs" / exp_id / "initialization.json").read_text())
        control = init["sector_control"]
        checks = validation["runs"][exp_id]
        assert checks["valid"] is True
        assert checks["shuffle_seed"] == 20241014 == control["shuffle_seed"]
        assert checks["eligible_tensors"] == 42
        assert control["all_multisets_preserved"] is True
        assert all(t["multiset_preserved"] for t in control["tensors"] if t["eligible"])
        assert checks["graft_relative_update"] >= 0.05
