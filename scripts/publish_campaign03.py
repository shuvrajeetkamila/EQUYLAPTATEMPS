#!/usr/bin/env python3
"""Build Campaign 3's central results from imported, hash-verified run records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("campaigns/campaign_03")
SCREEN = ROOT / "screen_38367618192"
CONFIRM = ROOT / "confirm_38371104410"
CONFIRM2 = ROOT / "confirm2_38375777144"


def load(path):
    return json.loads(Path(path).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def candidate(run_dir, exp_id):
    record = load(run_dir / "runs" / exp_id / "results.json")
    validation = record["validation"]
    return {
        "language_dev_nll": validation["groups"]["domain:language"]["token_nll"],
        "math_dev_nll": validation["groups"]["domain:math"]["token_nll"],
        "macro_dev_nll": macro(validation),
        "bypass_graft_macro_dev_nll": macro(record["ablations"]["bypass_graft"]),
        "zero_graft_macro_dev_nll": macro(record["ablations"]["zero_graft"]),
        "graft_relative_update": record.get("graft_delta", {}).get("mean_relative_update"),
        "reference_reconstruction_identical_logits": record.get(
            "reference_reconstruction_identical_logits"
        ),
        "trainable_parameters": record["structure"]["trainable_dense_seam_parameters"],
    }


def main():
    screen_promotion = load(SCREEN / "screen_promotion.json")
    native = load(SCREEN / "native_development.json")
    results = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "campaign": 3,
        "date": "2026-10-13",
        "parents": {
            "recipient": {
                "model_id": "state-spaces/mamba-130m-hf",
                "revision": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
            },
            "donor": {
                "model_id": "Qwen/Qwen2.5-Math-1.5B",
                "revision": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
            },
        },
        "screen": {
            "connected_run": "38367618192",
            "status": "COMPLETED",
            "selection_data": (
                "development only (16 WikiText-2 validation windows + 16 GSM8K TRAIN-held-out rows)"
            ),
            "host_language_dev_nll": native["recipient"]["groups"]["domain:language"]["token_nll"],
            "host_math_dev_nll": native["recipient"]["groups"]["domain:math"]["token_nll"],
            "donor_language_dev_nll_own_tokenizer": native["donor"]["groups"]["domain:language"][
                "token_nll"
            ],
            "donor_math_dev_nll_own_tokenizer": native["donor"]["groups"]["domain:math"][
                "token_nll"
            ],
            "candidates": {
                exp: candidate(SCREEN, exp)
                for exp in (
                    "c03_dense_cov_42",
                    "c03_conv_cov_42",
                    "c03_attn_cov_42",
                    "c03_gru_cov_42",
                )
            },
            "promotion": screen_promotion,
        },
        "diagnostic_confirm": {
            "connected_runs": ["38371104410", "38375777144"],
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "budget_by_content_macro_dev_nll": {
                "pretrained_32_steps": candidate(SCREEN, "c03_attn_cov_42")["macro_dev_nll"],
                "random_twin_32_steps": candidate(CONFIRM, "c03_attn_cov_42_randomtwin")[
                    "macro_dev_nll"
                ],
                "pretrained_96_steps": candidate(CONFIRM, "c03_attn_cov_42_long")["macro_dev_nll"],
                "random_twin_96_steps": candidate(CONFIRM2, "c03_attn_cov_42_long_randomtwin")[
                    "macro_dev_nll"
                ],
            },
            "seed_123_attention": candidate(CONFIRM, "c03_attn_cov_123"),
            "long_pretrained": candidate(CONFIRM, "c03_attn_cov_42_long"),
            "long_random_twin": candidate(CONFIRM2, "c03_attn_cov_42_long_randomtwin"),
        },
        "heldout_confirmation": {
            "status": "NOT_RUN",
            "reason": (
                "No candidate passed the frozen screen gates; per "
                "preregistration, held-out candidate evaluation is "
                "forbidden without promotion"
            ),
        },
        "outcome": {
            "classification": "C",
            "label": "Structural surgery without useful inheritance",
            "structural_transplantation": "PASS",
            "execution_and_reconstruction": "PASS",
            "host_language_retention_development": (
                "PASS for sequence candidates at screen budget (ratios "
                "1.416-1.444 vs 1.50 gate); not held-out confirmed"
            ),
            "functional_donor_contribution": (
                "FAIL (bypass equivalence at every budget; relative donor update ~0.09)"
            ),
            "pretrained_vs_random_sector": (
                "FAIL (random twin beats pretrained child at 32 and 96 steps)"
            ),
            "donor_math_retention": "FAIL (child math gains track the bypass path, not the donor)",
            "rationale": (
                "Sequence-aware bridges improve the child and pass "
                "development language retention, but the pretrained donor "
                "sector is a near-identity passenger: removing it costs "
                "nothing and replacing it with random blocks improves "
                "the child at both tested budgets."
            ),
        },
    }
    (ROOT / "campaign_03_results.json").write_text(json.dumps(results, indent=2) + "\n")
    ablation = {
        "kind": "development ablations from verified run records",
        "screen_run": "38367618192",
        "table": {
            exp: {
                "full_macro": results["screen"]["candidates"][exp]["macro_dev_nll"],
                "bypass_graft_macro": results["screen"]["candidates"][exp][
                    "bypass_graft_macro_dev_nll"
                ],
                "zero_graft_macro": results["screen"]["candidates"][exp][
                    "zero_graft_macro_dev_nll"
                ],
            }
            for exp in results["screen"]["candidates"]
        },
        "diagnostic_runs": ["38371104410", "38375777144"],
        "conclusion": (
            "bypass equivalence at 32 and 96 steps, pretrained and random: "
            "the donor sector's transformation is not used"
        ),
    }
    (ROOT / "campaign_03_ablation_results.json").write_text(json.dumps(ablation, indent=2) + "\n")
    print("wrote campaign_03_results.json and campaign_03_ablation_results.json")


if __name__ == "__main__":
    main()
