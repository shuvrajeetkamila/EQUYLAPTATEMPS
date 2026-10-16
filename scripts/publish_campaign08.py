#!/usr/bin/env python3
"""Build Campaign 8 published results + CSVs from imported raw run records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("campaigns/campaign_08")
C7 = Path("campaigns/campaign_07")
SOURCES = {
    "c07_dense_pre_42": (C7 / "screen_38542143640", "38542143640"),
    "c07_dense_rand_42": (C7 / "screen_38542143640", "38542143640"),
    "c07_dense_pre_123": (C7 / "replication_38544932532", "38544932532"),
    "c07_dense_rand_123": (C7 / "replication_38544932532", "38544932532"),
    "c08_calnoscale_pre_42": (ROOT / "stageD2_38563416962", "38563416962"),
    "c08_calnoscale_rand_42": (ROOT / "stageD2_38563416962", "38563416962"),
    "c08_calnoscale_pre_123": (ROOT / "stageF_38565839960", "38565839960"),
    "c08_calnoscale_rand_123": (ROOT / "stageF_38565839960", "38565839960"),
    "c08_calnoscale_pre_42_96": (ROOT / "stageEcal_38565837824", "38565837824"),
    "c08_calnoscale_rand_42_96": (ROOT / "stageEcal_38565837824", "38565837824"),
    "c08_orig_pre_42_96": (ROOT / "stageEorig_38566012225", "38566012225"),
    "c08_orig_rand_42_96": (ROOT / "stageEorig_38566012225", "38566012225"),
}
PAIRS = [
    ("original", 32, "42", "c07_dense_pre_42", "c07_dense_rand_42"),
    ("original", 32, "123", "c07_dense_pre_123", "c07_dense_rand_123"),
    ("calibrated_noscale", 32, "42", "c08_calnoscale_pre_42", "c08_calnoscale_rand_42"),
    ("calibrated_noscale", 32, "123", "c08_calnoscale_pre_123", "c08_calnoscale_rand_123"),
    ("original", 96, "42", "c08_orig_pre_42_96", "c08_orig_rand_42_96"),
    ("calibrated_noscale", 96, "42", "c08_calnoscale_pre_42_96", "c08_calnoscale_rand_42_96"),
]


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def main():
    runs = {}
    for exp_id, (folder, job) in SOURCES.items():
        record = json.loads((folder / "runs" / exp_id / "results.json").read_text())
        cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][exp_id]
        runs[exp_id] = {
            "job": job,
            "macro_dev_nll": macro(record["validation"]),
            "language_dev_nll": cand["language_dev_nll"],
            "math_dev_nll": cand["math_dev_nll"],
            "language_ratio_vs_host": cand["language_ratio_vs_host"],
            "bypass_macro": cand["bypass_macro"],
            "zero_macro": cand["zero_macro"],
            "graft_relative_update": cand["graft_relative_update"],
        }
    comparisons = []
    for condition, steps, seed, pre_id, twin_id in PAIRS:
        pre, twin = runs[pre_id], runs[twin_id]
        benefit = (twin["macro_dev_nll"] - pre["macro_dev_nll"]) / twin["macro_dev_nll"]
        comparisons.append(
            {
                "condition": condition,
                "steps": steps,
                "seed": seed,
                "pretrained_macro": pre["macro_dev_nll"],
                "twin_macro": twin["macro_dev_nll"],
                "benefit_fraction": benefit,
                "G_B_pass": pre["macro_dev_nll"] <= 0.98 * twin["macro_dev_nll"],
                "G_C_pass": pre["language_ratio_vs_host"] <= 1.50 and pre["macro_dev_nll"] <= 6.00,
            }
        )
    results = {
        "campaign": 8,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "pairing": "Mamba-130M host x SmolLM-135M [24,30) (identical to Campaign 7)",
        "calibration": (
            "covariance_orthogonal_noscale (Amendment 01); frozen "
            "covariance_orthogonal could not execute (scale-guard, 186x)"
        ),
        "host_language_dev_nll": 4.559452972412109,
        "runs": runs,
        "comparisons": comparisons,
        "promotion": False,
        "outcome": {
            "classification": "C",
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": (
                "Calibration dramatically improved the random twin (5.34/5.32 at 32 steps; "
                "4.06 at 96) but never the pretrained child: benefit was negative in all "
                "six paired comparisons (-13.8% to -118%). H1 and H2 rejected for this "
                "pairing; the experiment is valid, so Outcome C, not D."
            ),
        },
        "evidence_bundles_sha256": {
            "38561906471": "e16617dc545a9c5401dcd7e5395d8454c4a784e50d6ab4350a1c624bdc65f52d",
            "38563416962": "ed3140a89af739dca39f8db2225e12d34d90330619aba602e7f70d0a651670a8",
            "38565839960": "6b76ed18e4104e3c85d179a65c459a8556e645e6b34beb9d25609d40aa328b19",
            "38565837824": "2f142276c6529d8f49f269b7223d4fadb6f71c3d03091210511ae0acc6faca7f",
            "38566012225": "69eed38208fdc5e35cc3fc3780191fdb50e9279c3fee0cbc92bfb16ab23bc91d",
        },
    }
    (ROOT / "campaign_08_results.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(ROOT / "control_comparisons.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "steps",
                "seed",
                "pretrained_macro",
                "twin_macro",
                "benefit_percent",
                "G_B_pass",
                "G_C_pass",
            ]
        )
        for c in comparisons:
            writer.writerow(
                [
                    c["condition"],
                    c["steps"],
                    c["seed"],
                    f"{c['pretrained_macro']:.4f}",
                    f"{c['twin_macro']:.4f}",
                    f"{100 * c['benefit_fraction']:.2f}",
                    c["G_B_pass"],
                    c["G_C_pass"],
                ]
            )
    with open(ROOT / "donor_ablation_results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["experiment_id", "intact_macro", "bypass_macro", "zero_macro", "graft_relative_update"]
        )
        for exp_id, run in runs.items():
            writer.writerow(
                [
                    exp_id,
                    f"{run['macro_dev_nll']:.4f}",
                    f"{run['bypass_macro']:.4f}",
                    f"{run['zero_macro']:.4f}",
                    f"{run['graft_relative_update']:.3f}",
                ]
            )
    print("Campaign 8 results published; outcome:", results["outcome"]["classification"])


if __name__ == "__main__":
    main()
