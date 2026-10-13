"""Campaign 5 published results must match the imported run records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_05"


def load(*parts):
    return json.loads(ROOT.joinpath(*parts).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign05_results_trace_and_classification():
    results = load("campaign_05_results.json")
    assert results["outcome"]["classification"] == "C"
    assert results["heldout_confirmation"]["status"] == "NOT_RUN_NO_PROMOTION"
    screen = load("screen_38409870806", "runs", "c05_w6_late_42", "results.json")
    entry = results["candidates"]["c05_w6_late_42"]
    assert entry["pretrained"]["macro_dev_nll"] == macro(screen["validation"])
    assert entry["gates"]["B_pretrained_benefit_2pct"] is True
    assert entry["gates"]["D_useful_performance"] is False
    assert results["replication_w6_late_seed_123"]["benefit_fraction_vs_twin"] < 0
