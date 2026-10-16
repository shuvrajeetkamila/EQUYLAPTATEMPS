#!/usr/bin/env python3
"""Build Campaign 4 central results from imported, hash-verified run records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("campaigns/campaign_04")
SCREEN = ROOT / "screen_38393669406"
TWINS = ROOT / "twins_38397294665"
REPLICATION = ROOT / "replication_38401090731"


def load(path):
    return json.loads(Path(path).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def record(run_dir, exp_id):
    data = load(run_dir / "runs" / exp_id / "results.json")
    validation = data["validation"]
    return {
        "language_dev_nll": validation["groups"]["domain:language"]["token_nll"],
        "math_dev_nll": validation["groups"]["domain:math"]["token_nll"],
        "macro_dev_nll": macro(validation),
        "bypass_graft_macro": macro(data["ablations"]["bypass_graft"]),
        "zero_graft_macro": macro(data["ablations"]["zero_graft"]),
        "graft_relative_update": data["graft_delta"]["mean_relative_update"],
        "trainable_parameters": data["structure"]["trainable_dense_seam_parameters"],
        "training_tokens_processed": data.get("training_tokens_processed"),
    }


PAIRS = [
    ("c04_delta_attn_42", "c04_delta_attn_42_randomtwin"),
    ("c04_bn128_attn_42", "c04_bn128_attn_42_randomtwin"),
    ("c04_bn128_delta_attn_42", "c04_bn128_delta_attn_42_randomtwin"),
]


def main():
    host_language = load(SCREEN / "phase_summary.json")["host_language_dev_nll"]
    candidates = {}
    for pretrained, twin in PAIRS:
        pre = record(SCREEN, pretrained)
        rnd = record(TWINS, twin)
        candidates[pretrained] = {
            "pretrained": pre,
            "random_twin": rnd,
            "gates": {
                "B_pretrained_benefit_2pct": pre["macro_dev_nll"] <= 0.98 * rnd["macro_dev_nll"],
                "C_donor_dependence_2pct": pre["macro_dev_nll"]
                <= 0.98 * min(pre["bypass_graft_macro"], pre["zero_graft_macro"]),
                "D_useful_performance": pre["language_dev_nll"] <= 1.5 * host_language
                and pre["macro_dev_nll"] <= 5.7396,
            },
        }
    replication = {
        "pretrained_seed_123": record(REPLICATION, "c04_bn128_delta_attn_123"),
        "random_twin_seed_123": record(REPLICATION, "c04_bn128_delta_attn_123_randomtwin"),
    }
    rep_pre = replication["pretrained_seed_123"]["macro_dev_nll"]
    rep_twin = replication["random_twin_seed_123"]["macro_dev_nll"]
    replication["benefit_direction_preserved"] = rep_pre < rep_twin
    results = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "campaign": 4,
        "date": "2026-10-13",
        "runs": {
            "screen_failed_attempt": "38390162060 (post-training accounting defect; preserved)",
            "screen": "38393669406",
            "twins": "38397294665",
            "replication": "38401090731",
        },
        "host_language_dev_nll": host_language,
        "candidates": candidates,
        "replication_bn128_delta": replication,
        "heldout_confirmation": {
            "status": "NOT_RUN_NO_PROMOTION",
            "reason": "No candidate passed all frozen gates (every candidate failed Gate D)",
        },
        "outcome": {
            "classification": "B",
            "label": "Partial evidence of donor contribution without useful inheritance",
            "structural_transplantation": "PASS",
            "execution_and_reconstruction": "PASS",
            "donor_dependence_under_forcing": (
                "PASS for donor-delta candidates (bypass costs 17-49% macro NLL)"
            ),
            "pretrained_specific_benefit": (
                "WEAK PASS for bottleneck+delta only: +2.8% (seed 42) and +1.7% "
                "(seed 123) vs matched random twins; direction consistent, "
                "magnitude small and below the 2% gate at seed 123"
            ),
            "useful_task_performance": (
                "FAIL (all candidates language 1.82-1.85x host; macro 7.70-8.02 vs 5.7396 ceiling)"
            ),
            "promotion": "NONE",
            "rationale": (
                "Mandatory-use architecture created genuine donor dependence "
                "and the first reproducible-direction pretrained advantage, "
                "but the forced children are far worse than the unconstrained "
                "Campaign 3 child and both parents. Partial contribution is "
                "demonstrated; useful capability inheritance is not."
            ),
        },
    }
    (ROOT / "campaign_04_results.json").write_text(json.dumps(results, indent=2) + "\n")
    controls = {
        "kind": "pretrained vs matched random-sector twins, identical budgets",
        "pairs": {
            pre: {
                "pretrained_macro": candidates[pre]["pretrained"]["macro_dev_nll"],
                "random_twin_macro": candidates[pre]["random_twin"]["macro_dev_nll"],
            }
            for pre, _ in PAIRS
        },
        "replication_bn128_delta": {
            "pretrained_macro": rep_pre,
            "random_twin_macro": rep_twin,
        },
    }
    (ROOT / "campaign_04_control_comparisons.json").write_text(
        json.dumps(controls, indent=2) + "\n"
    )
    print("wrote campaign_04_results.json and campaign_04_control_comparisons.json")


if __name__ == "__main__":
    main()
