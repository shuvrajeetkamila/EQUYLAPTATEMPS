"""Bounded math diagnostic consistency fixtures; no pretrained benchmark claims."""

import json

import pytest

from equylaptatemps.audit import sha256_file
from equylaptatemps.completion import summarize_completed_math
from equylaptatemps.math_evidence import registered_prompt, validate_completed_math

VARIANTS = [
    "recipient_parent",
    "donor_parent",
    "published_calibrated_child",
    "uncalibrated_child",
    "random_interface",
    "matched_random_sector",
    "bypass_graft",
    "bypass_prefix",
    "bypass_suffix",
    "zero_input_seam",
    "zero_output_seam",
]


def fixture_record(root):
    config = {
        "variants": VARIANTS,
        "benchmark": {"sample_count": 1, "max_new_tokens": 4, "max_input_tokens": 64},
        "inheritance": {"status": "NOT_RUN", "reason": "Unit schema fixture"},
    }
    (root / "registration.json").write_text(json.dumps(config))
    (root / "data-selection.json").write_text(json.dumps({"test_ids": [0]}))
    row = {
        "id": 0,
        "question_sha256": "fixture",
        "prompt_sha256": "fixture",
        "reference": "#### 2",
        "response": "#### 2\n",
        "stop_reason": "completed_answer",
        "generated_ids": [8, 9],
        "input_tokens": 8,
        "prediction": "2",
        "correct": True,
    }
    result = {
        "status": "PASS",
        "registration_sha256": sha256_file(root / "registration.json"),
        "comparisons": {},
        "cache_oracles": {name: {"status": "PASS", "argmax_equal": True} for name in VARIANTS},
        "reverse_cache_oracle": {"status": "PASS"},
        "inheritance": config["inheritance"],
        "frozen_invariance": {"status": "PASS", "originals_unchanged": True},
        "fresh_process": {"generation_identical": True, "parents_loaded": False},
    }
    for name in VARIANTS:
        file = root / f"{name}.jsonl"
        file.write_text(json.dumps(row) + "\n")
        result["comparisons"][name] = {
            "status": "PASS",
            "kind": "measured",
            "records": file.name,
            "records_sha256": sha256_file(file),
            "metrics": summarize_completed_math([row]),
        }
    (root / "results.json").write_text(json.dumps(result))
    return result


def test_math_registration_uses_human_fewshot_and_final_marker_prompt():
    prompt = registered_prompt(
        "Three plus two?", [{"question": "One plus one?", "answer": "#### 2"}]
    )
    assert "One plus one?\nAnswer: #### 2" in prompt
    assert prompt.endswith("Three plus two?\nAnswer: ")


def test_complete_record_is_consistent_not_inheritance(tmp_path):
    fixture_record(tmp_path)
    assert validate_completed_math(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize(
    "case",
    ["invented_metric", "missing_variant", "failed_oracle", "inheritance", "fresh_parent_loading"],
)
def test_math_diagnostic_cannot_fake_scores_or_pass_failed_oracles(tmp_path, case):
    result = fixture_record(tmp_path)
    if case == "invented_metric":
        result["comparisons"]["donor_parent"]["metrics"]["accuracy"] = 0.5
    elif case == "missing_variant":
        result["comparisons"].pop("matched_random_sector")
    elif case == "failed_oracle":
        result["cache_oracles"]["donor_parent"]["status"] = "FAIL"
    elif case == "inheritance":
        result["inheritance"] = {"status": "PASS"}
    else:
        result["fresh_process"]["parents_loaded"] = True
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        validate_completed_math(tmp_path)
