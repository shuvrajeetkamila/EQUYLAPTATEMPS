"""Campaign 16 frozen rules, raw-record traceability, C15 reproduction."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publish_campaign16 import decide, macro, ranking  # noqa: E402

C16 = ROOT / "campaigns" / "campaign_16"
PANELS = {
    "low_0.3x": ("lrlow_38882548685", "0.3x"),
    "base_1x": ("lrbase_38885462762", "1x"),
    "high_3x": ("lrhigh_38888327387", "3x"),
}
SEEDS = (42, 123, 31415)


def _record(lr, condition, seed):
    panel_dir, tag = PANELS[lr]
    path = C16 / panel_dir / "runs" / f"c16_{condition}_lr{tag}_s{seed}" / "results.json"
    return json.loads(path.read_text())


def _evals(record):
    return {e["step"]: macro(e["validation"]) for e in record["checkpoint_evaluations"]}


def test_ranking_and_decision_rules_hand_computed():
    assert ranking([1.0, 0.5, -0.1]) == "NEW"
    assert ranking([-1.0, -0.5, 0.1]) == "OLD"
    assert ranking([0.3, 0.1, 0.05]) == "TIE"  # mean below the 0.25 threshold
    all_tie = {(lr, t): "TIE" for lr in PANELS for t in (32, 96)}
    assert decide(True, True, True, all_tie, False) == "NO_RELIABLE_DIFFERENCE"
    assert decide(False, True, True, all_tie, False) == "IMPLEMENTATION_FAILURE"
    assert decide(True, True, False, all_tie, False) == "INCONCLUSIVE"


def test_matrix_complete_and_lr_monotone_pattern():
    for lr in PANELS:
        for seed in SEEDS:
            for condition in ("old", "new"):
                rec = _record(lr, condition, seed)
                assert rec["training"]["execution_status"] == "COMPLETED"
                assert rec["parent_weights_frozen"] is True
                assert rec["structure"]["trainable_dense_seam_parameters"] == 667008
    import statistics

    mean32 = {}
    for lr in PANELS:
        deltas = [
            _evals(_record(lr, "old", s))[32] - _evals(_record(lr, "new", s))[32] for s in SEEDS
        ]
        mean32[lr] = statistics.fmean(deltas)
    # The frozen finding: monotone LR response of the paired difference.
    assert mean32["low_0.3x"] < mean32["base_1x"] < mean32["high_3x"]
    assert mean32["low_0.3x"] < -2.0  # old connector far ahead at low LR
    analysis = json.loads((C16 / "paired_analysis.json").read_text())
    assert analysis["outcome"] == "LEARNING_RATE_SENSITIVE"
    assert analysis["mechanism_supported"] is True


def test_baseline_cells_reproduce_campaign15():
    c15_panels = {
        42: "panel42_38865626143",
        123: "panel123_38865628701",
        31415: "panel31415_38865631168",
    }
    for seed, d in c15_panels.items():
        for condition in ("old", "new"):
            old_rec = json.loads(
                (
                    ROOT
                    / "campaigns"
                    / "campaign_15"
                    / d
                    / "runs"
                    / f"c15_{condition}_{seed}"
                    / "results.json"
                ).read_text()
            )
            new_rec = _record("base_1x", condition, seed)
            for step in (0, 32, 96):
                assert abs(_evals(old_rec)[step] - _evals(new_rec)[step]) <= 0.01
