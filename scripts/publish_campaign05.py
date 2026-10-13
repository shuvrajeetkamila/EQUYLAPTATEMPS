#!/usr/bin/env python3
"""Build Campaign 5 central results from imported, hash-verified run records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("campaigns/campaign_05")
SCREEN = ROOT / "screen_38409870806"
TWINS = ROOT / "twins_38414175865"
POSITIONS = ROOT / "positions_38418551528"
REPLICATION = ROOT / "replication_38422937098"


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
    }


def pair(pretrained_dir, pretrained, twin_dir, twin):
    pre, rnd = record(pretrained_dir, pretrained), record(twin_dir, twin)
    return {
        "pretrained": pre,
        "random_twin": rnd,
        "benefit_fraction_vs_twin": (rnd["macro_dev_nll"] - pre["macro_dev_nll"])
        / rnd["macro_dev_nll"],
        "gates": {
            "B_pretrained_benefit_2pct": pre["macro_dev_nll"] <= 0.98 * rnd["macro_dev_nll"],
            "C_donor_dependence_2pct": pre["macro_dev_nll"]
            <= 0.98 * min(pre["bypass_graft_macro"], pre["zero_graft_macro"]),
            "D_useful_performance": False,
        },
    }


def main():
    host_language = load(SCREEN / "phase_summary.json")["host_language_dev_nll"]
    candidates = {
        "c05_w1_late_42": pair(SCREEN, "c05_w1_late_42", TWINS, "c05_w1_late_42_randomtwin"),
        "c05_w4_late_42": pair(SCREEN, "c05_w4_late_42", TWINS, "c05_w4_late_42_randomtwin"),
        "c05_w6_late_42": pair(SCREEN, "c05_w6_late_42", TWINS, "c05_w6_late_42_randomtwin"),
        "c05_w4_early_42": pair(
            POSITIONS, "c05_w4_early_42", POSITIONS, "c05_w4_early_42_randomtwin"
        ),
        "c05_w4_mid_42": pair(POSITIONS, "c05_w4_mid_42", POSITIONS, "c05_w4_mid_42_randomtwin"),
    }
    for entry in candidates.values():
        entry["gates"]["D_useful_performance"] = (
            entry["pretrained"]["language_dev_nll"] <= 1.5 * host_language
            and entry["pretrained"]["macro_dev_nll"] <= 5.7396
        )
    rep = pair(REPLICATION, "c05_w6_late_123", REPLICATION, "c05_w6_late_123_randomtwin")
    results = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "campaign": 5,
        "date": "2026-10-13",
        "runs": {
            "screen": "38409870806",
            "twins": "38414175865",
            "positions": "38418551528",
            "replication": "38422937098",
        },
        "host_language_dev_nll": host_language,
        "campaign_04_reference_w2_late": {
            "pretrained_macro": 7.7873,
            "random_twin_macro": 8.0091,
            "benefit_fraction_vs_twin": 0.0277,
            "source": "Campaign 4 verified records (not rerun)",
        },
        "candidates": candidates,
        "replication_w6_late_seed_123": rep,
        "heldout_confirmation": {
            "status": "NOT_RUN_NO_PROMOTION",
            "reason": (
                "Every candidate failed Gate D; the largest Gate-B benefit reversed at seed 123"
            ),
        },
        "outcome": {
            "classification": "C",
            "label": ("Structural surgery without useful inheritance (width/placement hypothesis)"),
            "donor_dependence_under_forcing": (
                "PASS (removal/neutralization catastrophic at every sector)"
            ),
            "pretrained_specific_benefit": (
                "FAIL as a stable effect: benefits vs twins +0.5%, -0.7%, +4.5% "
                "(widths 1/4/6 late), -1.6%/-0.5% (width-4 early/middle), and "
                "the +4.5% reversed to -2.1% at seed 123"
            ),
            "useful_task_performance": ("FAIL (all language 1.81-1.87x host; macro 7.79-8.16)"),
            "width_scaling": (
                "Donor relative update grows with width (0.05/0.13/0.18) but task benefit does not"
            ),
            "promotion": "NONE",
            "rationale": (
                "Placement and width do not rescue inheritance: donor "
                "dependence is guaranteed by the delta architecture, yet "
                "pretrained-specific advantage is indistinguishable from "
                "seed noise across six comparisons and fails absolute "
                "performance everywhere. Campaign 4's two-seed w2 benefit "
                "remains an isolated observation, not a scalable effect."
            ),
        },
    }
    (ROOT / "campaign_05_results.json").write_text(json.dumps(results, indent=2) + "\n")
    controls = {
        "kind": "pretrained vs matched random twins by sector",
        "benefit_fraction_vs_twin": {
            name: entry["benefit_fraction_vs_twin"] for name, entry in candidates.items()
        },
        "replication_w6_seed_123": rep["benefit_fraction_vs_twin"],
        "campaign_04_reference_w2": 0.0277,
    }
    (ROOT / "campaign_05_control_comparisons.json").write_text(
        json.dumps(controls, indent=2) + "\n"
    )
    print("wrote campaign_05_results.json and campaign_05_control_comparisons.json")


if __name__ == "__main__":
    main()
