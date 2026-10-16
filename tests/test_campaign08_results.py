"""Campaign 8 published results must match the imported raw records."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "campaigns"
JOBS = {
    "c07_dense_pre_42": ("campaign_07/screen_38542143640",),
    "c08_calnoscale_pre_42": ("campaign_08/stageD2_38563416962",),
    "c08_calnoscale_rand_42_96": ("campaign_08/stageEcal_38565837824",),
    "c08_orig_rand_42_96": ("campaign_08/stageEorig_38566012225",),
    "c08_calnoscale_pre_123": ("campaign_08/stageF_38565839960",),
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def test_campaign08_outcome_traces_to_records():
    results = json.loads((ROOT / "campaign_08" / "campaign_08_results.json").read_text())
    assert results["outcome"]["classification"] == "C"
    assert results["outcome"]["held_out_status"] == "NOT_RUN_NO_PROMOTION"
    assert results["promotion"] is False
    for exp_id, (folder,) in JOBS.items():
        record = json.loads((ROOT / folder / "runs" / exp_id / "results.json").read_text())
        assert results["runs"][exp_id]["macro_dev_nll"] == macro(record["validation"])
    # The campaign's central fact: benefit was negative in every paired comparison.
    assert len(results["comparisons"]) == 6
    assert all(c["benefit_fraction"] < 0 and not c["G_B_pass"] for c in results["comparisons"])
    # And calibration's twin improvement is real (it is what makes the design informative).
    runs = results["runs"]
    assert (
        runs["c08_calnoscale_rand_42"]["macro_dev_nll"] < runs["c07_dense_rand_42"]["macro_dev_nll"]
    )
