"""Diagnostic schema/unit observations, not pretrained capability measurements."""

import json

import pytest

from equylaptatemps.audit import sha256_file
from equylaptatemps.benchmarks import summarize_language
from equylaptatemps.diagnostics import validate_language_diagnostic
from equylaptatemps.evidence import paired_interval

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


def fixture_records(root):
    config = {
        "variants": VARIANTS,
        "benchmark": {"window_count": 2},
        "seed": 42,
        "criteria": {
            "minimum_windows": 2,
            "maximum_language_bpb": 3.0,
            "language_retention_ratio": 1.1,
        },
    }
    (root / "registration.json").write_text(json.dumps(config))
    rows = [
        {"id": i, "text_sha256": str(i), "nll_sum": 8.0, "text_bytes": 8, "target_tokens": 2}
        for i in range(2)
    ]
    result = {
        "status": "PASS",
        "registration_sha256": sha256_file(root / "registration.json"),
        "comparisons": {},
        "paired_language": {},
        "language_retention": {"status": "PASS", "child_to_recipient_bpb_ratio": 1},
        "learned_sector_advantage": {"status": "FAIL"},
        "inheritance": {"status": "NOT_RUN"},
        "math_benchmark": {"status": "NOT_RUN"},
        "frozen_invariance": {"status": "PASS", "original": True},
        "fresh_reference": {"status": "PASS", "identical_logits": True, "generation_equal": True},
    }
    for name in VARIANTS:
        file = root / f"{name}.jsonl"
        file.write_text("".join(json.dumps(r) + "\n" for r in rows))
        result["comparisons"][name] = {
            "status": "PASS",
            "kind": "measured",
            "metrics": summarize_language(rows),
            "records": file.name,
            "records_sha256": sha256_file(file),
        }
    for name in ("matched_random_sector", "bypass_graft", "bypass_prefix", "bypass_suffix"):
        result["paired_language"][name] = paired_interval(rows, rows, "wikitext2", seed=42)
    (root / "results.json").write_text(json.dumps(result))
    return result


def test_complete_language_diagnostic_is_consistent_not_inheritance(tmp_path):
    fixture_records(tmp_path)
    assert validate_language_diagnostic(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation", ["missing_variant", "inheritance_claim", "metric", "retention", "paired"]
)
def test_diagnostic_rejects_fabricated_or_missing_summaries(tmp_path, mutation):
    result = fixture_records(tmp_path)
    if mutation == "missing_variant":
        result["comparisons"].pop("matched_random_sector")
    elif mutation == "inheritance_claim":
        result["inheritance"]["status"] = "PASS"
    elif mutation == "metric":
        result["comparisons"]["published_calibrated_child"]["metrics"]["bits_per_byte"] = 0
    elif mutation == "retention":
        result["language_retention"]["status"] = "FAIL"
    else:
        result["paired_language"]["matched_random_sector"]["difference"] = 1
    (tmp_path / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        validate_language_diagnostic(tmp_path)
