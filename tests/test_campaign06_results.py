"""Campaign 6 published results must match the imported run records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_06"


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign06_capacity_decision_traces_to_records():
    results = json.loads((ROOT / "campaign_06_results.json").read_text())
    assert results["decision"]["capacity_remediable"] is False
    assert results["outcome"]["classification"] == "C"
    record = json.loads(
        (
            ROOT / "capacity_38428594451" / "runs" / "c06_bn256_delta_rand_42" / "results.json"
        ).read_text()
    )
    assert results["new_runs"]["c06_bn256_delta_rand_42"]["macro_dev_nll"] == macro(
        record["validation"]
    )
    assert (
        results["new_runs"]["c06_bn256_delta_rand_42"]["macro_dev_nll"]
        > results["decision"]["bn256_32_threshold"]
    )
