"""Campaign 3 published results must match the imported run records exactly."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns" / "campaign_03"


def load(*parts):
    return json.loads(ROOT.joinpath(*parts).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign03_results_trace_to_run_records():
    results = load("campaign_03_results.json")
    screen = load("screen_38367618192", "screen_promotion.json")
    assert results["screen"]["promotion"] == screen
    assert results["outcome"]["classification"] == "C"
    assert results["heldout_confirmation"]["status"] == "NOT_RUN"
    for exp, values in results["screen"]["candidates"].items():
        record = load("screen_38367618192", "runs", exp, "results.json")
        assert values["macro_dev_nll"] == macro(record["validation"])
        assert values["bypass_graft_macro_dev_nll"] == macro(record["ablations"]["bypass_graft"])


def test_campaign03_random_twin_beats_pretrained_at_both_budgets():
    table = load("campaign_03_results.json")["diagnostic_confirm"][
        "budget_by_content_macro_dev_nll"
    ]
    assert table["random_twin_32_steps"] < table["pretrained_32_steps"]
    assert table["random_twin_96_steps"] < table["pretrained_96_steps"]
