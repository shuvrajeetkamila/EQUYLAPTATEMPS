"""Campaign 13 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_13"
JOBS = {32: "screen32_38767422815", 96: "budget96_38770695036"}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def _eid(cond, seed, steps):
    return f"c13_{cond}_{seed}" + ("_96" if steps == 96 else "")


def test_campaign13_outcome_c_traces_to_all_records():
    results = json.loads((ROOT / "results.json").read_text())
    assert results["outcome"]["classification"] == "C"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    assert results["positive_control_pass"] is False
    assert results["control_valid"] is True
    assert results["arrangement_sensitivity"] == "none"
    assert len(results["runs"]) == 21  # 20 confirmatory + F self-reconstruction
    for steps, seeds in ((32, (42, 123, 31415)), (96, (42, 123))):
        for cond in ("pre", "row", "full", "rand"):
            for seed in seeds:
                eid = _eid(cond, seed, steps)
                record = json.loads(
                    (ROOT / JOBS[steps] / "runs" / eid / "results.json").read_text()
                )
                assert results["runs"][eid]["macro_dev_nll"] == macro(record["validation"])
                assert record["training"]["parent_weights_unchanged"] is True
                assert record["test_metrics"] is None
                assert record["structure"]["positive_control"] is True
    primary = {c["seed"]: c for c in results["comparisons"] if c["steps"] == 32}
    assert primary[42]["vs_B_percent"] < 2.0  # gate fails at seed 42 margins
    assert primary[123]["vs_B_percent"] >= 1.0 and primary[123]["vs_C_percent"] >= 1.0
    assert primary[31415]["vs_C_percent"] < 0  # and at seed 31415 vs full scramble
    # F is mechanical only: present, labeled, and far worse than the native host.
    assert results["runs"]["c13_self_42"]["condition"] == "F"
    assert results["runs"]["c13_self_42"]["macro_dev_nll"] > results["host_macro_dev_nll"]


def test_campaign13_control_validity_traces_to_records():
    doc = json.loads((ROOT / "control_validation.json").read_text())
    assert doc["pre_training_valid"] is True
    post = doc["post_training_run_checks"]
    assert post["control_valid"] is True
    assert len(post["runs"]) == 10
    for eid, checks in post["runs"].items():
        assert checks["valid"] is True
        assert checks["eligible_tensors"] == 42
        assert checks["ablations_finite_and_above_intact"] is True
        folder = JOBS[96] if eid.endswith("_96") else JOBS[32]
        init = json.loads((ROOT / folder / "runs" / eid / "initialization.json").read_text())
        assert init["sector_control"]["all_multisets_preserved"] is True
