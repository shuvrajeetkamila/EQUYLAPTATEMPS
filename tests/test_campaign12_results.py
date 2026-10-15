"""Campaign 12 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_12"
JOBS = {32: ("screen32_38726922262",), 96: ("budget96_38738479882",)}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def _eid(cond, seed, steps):
    return f"c12_{cond}_{seed}" + ("_96" if steps == 96 else "")


def test_campaign12_outcome_c_traces_to_all_20_records():
    results = json.loads((ROOT / "results.json").read_text())
    assert results["outcome"]["classification"] == "C"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    assert results["control_valid"] is True
    assert results["learned_structure_gate_pass"] is False
    assert len(results["runs"]) == 20
    for steps, seeds in ((32, (42, 123, 31415)), (96, (42, 123))):
        folder = JOBS[steps][0]
        for cond in ("pre", "row", "full", "rand"):
            for seed in seeds:
                eid = _eid(cond, seed, steps)
                record = json.loads((ROOT / folder / "runs" / eid / "results.json").read_text())
                assert results["runs"][eid]["macro_dev_nll"] == macro(record["validation"])
                assert record["training"]["parent_weights_unchanged"] is True
                assert record["test_metrics"] is None  # held-out never opened
    primary = {c["seed"]: c for c in results["comparisons"] if c["steps"] == 32}
    assert primary[42]["vs_B_percent"] >= 2.0 and primary[42]["vs_C_percent"] >= 2.0
    assert primary[123]["vs_C_percent"] < 1.0  # gate fails at seed 123
    assert primary[31415]["vs_B_percent"] < 0 and primary[31415]["vs_C_percent"] < 0


def test_campaign12_control_validity_traces_to_records():
    doc = json.loads((ROOT / "control_validation.json").read_text())
    assert doc["pre_training_valid"] is True
    post = doc["post_training_run_checks"]
    assert post["control_valid"] is True
    assert len(post["runs"]) == 10  # 5 within-row + 5 full-tensor runs
    ordering_exceptions = 0
    for eid, checks in post["runs"].items():
        assert checks["valid"] is True
        assert checks["eligible_tensors"] == 42
        assert checks["graft_update_at_least_0_05"] is True
        if not checks["bypass_below_zero_ordering"]:
            ordering_exceptions += 1
        init = json.loads(
            (
                ROOT
                / (JOBS[96][0] if eid.endswith("_96") else JOBS[32][0])
                / "runs"
                / eid
                / "initialization.json"
            ).read_text()
        )
        assert init["sector_control"]["all_multisets_preserved"] is True
    # Amendment 01: the two bypass>zero exceptions are reported, not hidden.
    assert ordering_exceptions == 2
