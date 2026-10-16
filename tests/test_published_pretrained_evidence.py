"""Recompute the published real pilot, offline and without downloading model/dataset weights."""

import json
from pathlib import Path

from equylaptatemps.evidence import validate_results

ROOT = Path(__file__).resolve().parents[1]


def test_published_actual_pilot_metric_records_and_config_are_consistent():
    path = ROOT / "reports/pretrained/run-38293921543"
    assert validate_results(path)["status"] == "PASS"
    result = json.loads((path / "results.json").read_text())
    assert result["source_commit"] == "95b2831a7f54ae7755e6892e364d993a779a3193"
    assert result["comparisons"]["calibrated_child"]["gsm8k"]["metrics"]["correct"] == 0
    assert result["search"]["selected"] == [20, 22]
    assert result["stages"]["reload"]["identical_logits"] is True
    assert result["frozen_invariance"]["all_original_parent_tensors_unchanged"] is True
    assert result["inheritance"]["retention"]["status"] == "NOT_RUN"
    assert result["inheritance"]["complementarity"]["status"] == "NOT_RUN"


def test_failed_real_attempt_is_preserved_and_not_rebranded_complete():
    path = ROOT / "reports/pretrained/attempt-38292930207"
    assert validate_results(path)["status"] == "PASS"
    result = json.loads((path / "results.json").read_text())
    assert result["error"]["type"] == "AttributeError"
    assert result["stages"]["parent_baselines"]["status"] == "FAIL"
    assert result["comparisons"]["calibrated_child"]["gsm8k"]["metrics"] is None
