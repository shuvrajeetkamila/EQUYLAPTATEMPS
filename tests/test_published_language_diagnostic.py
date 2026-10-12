"""Offline recomputation of actual remote diagnostic evidence, not a new inference benchmark."""

import json
from pathlib import Path

from equylaptatemps.evidence import validate_results

ROOT = Path(__file__).resolve().parents[1] / "reports/pretrained/run-38301210044"


def test_actual_diagnostic_raw_records_support_negative_registered_verdicts():
    assert validate_results(ROOT)["status"] == "PASS"
    result = json.loads((ROOT / "results.json").read_text())
    assert result["source_commit"] == "7ca1a4dba38ec663065c8300601e8d44148b0758"
    assert result["language_retention"]["status"] == "FAIL"
    assert result["learned_sector_advantage"]["status"] == "FAIL"
    assert result["paired_language"]["matched_random_sector"]["bootstrap_95"][1] < 0
    assert result["math_benchmark"]["status"] == "NOT_RUN"
    assert result["inheritance"]["status"] == "NOT_RUN"


def test_actual_reference_process_and_control_training_are_explicit():
    result = json.loads((ROOT / "results.json").read_text())
    training = json.loads((ROOT / "control-calibration.json").read_text())
    assert result["fresh_reference"]["identical_logits"]
    assert result["fresh_reference"]["generation_equal"]
    assert training["parent_kind"] == "random_sector_control"
    assert training["steps"] == 16
    assert len(training["trainable_tensors"]) == 8
    assert all(
        value is True for key, value in result["frozen_invariance"].items() if key != "status"
    )
