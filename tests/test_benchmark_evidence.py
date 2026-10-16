"""Scoring/integrity unit fixtures only, never real benchmark performance claims."""

import json
import math

import pytest
import torch

from equylaptatemps.audit import sha256_file
from equylaptatemps.benchmarks import (
    normalize_number,
    numeric_answer,
    summarize_language,
    summarize_math,
    wilson,
)
from equylaptatemps.evidence import comparison_template, paired_interval, pending, validate_results


@pytest.mark.parametrize(
    "text,answer",
    [
        ("#### 1,250.00", "1.25E+3"),
        (r"Thus \boxed{-2.5}", "-2.5"),
        ("three calculations; final value 42.", "42"),
        ("No numerical answer.", None),
    ],
)
def test_numeric_normalization(text, answer):
    assert numeric_answer(text) == answer


def test_strict_and_flexible_scoring_and_confidence():
    assert normalize_number("NaN") is None
    assert numeric_answer("the answer is 2", strict=True) is None
    rows = [
        {"reference": "One plus one. #### 2", "response": "2", "prediction": "2", "correct": True}
    ]
    summary = summarize_math(rows)
    assert summary["accuracy"] == 1 and summary["strict_accuracy"] == 0
    assert summary["wilson_95"][0] < 0.3
    assert wilson(0, 0) is None
    with pytest.raises(ValueError, match="Missing GSM"):
        summarize_math([])
    rows[0]["correct"] = False
    with pytest.raises(ValueError, match="substituted"):
        summarize_math(rows)


def test_byte_normalized_language_is_not_cross_vocab_perplexity():
    rows = [{"nll_sum": 8 * math.log(2), "target_tokens": 2, "text_bytes": 8}]
    summary = summarize_language(rows)
    assert summary["bits_per_byte"] == 1
    assert summary["target_tokens"] == 2
    with pytest.raises(ValueError):
        summarize_language([{"nll_sum": float("nan"), "target_tokens": 2, "text_bytes": 8}])


def prepare_evidence(tmp_path):
    registration = tmp_path / "registration.json"
    registration.write_text("{}\n")
    result = {
        "registration_file": "registration.json",
        "registration_sha256": sha256_file(registration),
        "stages": {"benchmarks": pending()},
        "comparisons": comparison_template(),
        "sample_counts": {"gsm8k": 1, "wikitext2": 1},
    }
    (tmp_path / "results.json").write_text(json.dumps(result))
    return result


def test_unrun_is_null_and_not_optimistically_zero(tmp_path):
    result = prepare_evidence(tmp_path)
    assert validate_results(tmp_path)["status"] == "PASS"
    result["comparisons"]["calibrated_child"]["gsm8k"]["metrics"] = {"accuracy": 0}
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="must not claim"):
        validate_results(tmp_path)


def test_incomplete_experiment_cannot_mark_benchmarks_pass(tmp_path):
    result = prepare_evidence(tmp_path)
    result["stages"]["benchmarks"] = {"status": "PASS"}
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_results(tmp_path)


def test_tampered_or_invented_metric_summary_rejected(tmp_path):
    result = prepare_evidence(tmp_path)
    rows = [
        {"id": 0, "reference": "#### 2", "response": "#### 3", "prediction": "3", "correct": False}
    ]
    observations = tmp_path / "observations.jsonl"
    observations.write_text(json.dumps(rows[0]) + "\n")
    entry = {
        "status": "PASS",
        "kind": "measured",
        "records": "observations.jsonl",
        "records_sha256": sha256_file(observations),
        "metrics": summarize_math(rows),
    }
    result["comparisons"]["calibrated_child"]["gsm8k"] = entry
    (tmp_path / "results.json").write_text(json.dumps(result))
    assert validate_results(tmp_path)["status"] == "PASS"
    entry["metrics"]["accuracy"] = 1.0
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Fabricated"):
        validate_results(tmp_path)
    entry["metrics"]["accuracy"] = 0
    observations.write_text(observations.read_text() + "\n")
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="checksum"):
        validate_results(tmp_path)


def test_paired_intervals_align_real_record_ids():
    a = [{"id": i, "correct": True} for i in range(4)]
    b = [{"id": i, "correct": False} for i in range(4)]
    assert paired_interval(a, b, "gsm8k", repetitions=100)["bootstrap_95"] == [1, 1]
    b[0]["id"] = 20
    with pytest.raises(ValueError, match="identical ordered"):
        paired_interval(a, b, "gsm8k")


@pytest.mark.parametrize(
    "ablation", ["bypass_prefix", "bypass_suffix", "zero_input_seam", "zero_output_seam"]
)
def test_recipient_and_interface_ablations_are_evaluation_only(child_bundle, ablation):
    child, *_ = child_bundle
    ids = torch.tensor([[8, 9, 10]])
    child.eval()
    output = child(ids, ablation=ablation).logits
    assert torch.isfinite(output).all()
    child.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        child(ids, ablation=ablation)
