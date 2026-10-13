"""Campaign 4 published results must match the imported run records exactly."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_04"


def load(*parts):
    return json.loads(ROOT.joinpath(*parts).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign04_results_trace_to_records():
    results = load("campaign_04_results.json")
    assert results["outcome"]["classification"] == "B"
    assert results["heldout_confirmation"]["status"] == "NOT_RUN_NO_PROMOTION"
    for exp, entry in results["candidates"].items():
        screen = load("screen_38393669406", "runs", exp, "results.json")
        assert entry["pretrained"]["macro_dev_nll"] == macro(screen["validation"])
        assert entry["gates"]["D_useful_performance"] is False
    bn = results["candidates"]["c04_bn128_delta_attn_42"]
    assert bn["gates"]["B_pretrained_benefit_2pct"] is True
    assert bn["gates"]["C_donor_dependence_2pct"] is True
    assert results["replication_bn128_delta"]["benefit_direction_preserved"] is True
