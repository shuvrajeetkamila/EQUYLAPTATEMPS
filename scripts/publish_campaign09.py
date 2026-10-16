#!/usr/bin/env python3
"""Build Campaign 9 published results from imported raw run records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("campaigns/campaign_09")
C7 = Path("campaigns/campaign_07")
SOURCES = {
    "c07_dense_pre_42": (C7 / "screen_38542143640", "38542143640", "original"),
    "c07_dense_rand_42": (C7 / "screen_38542143640", "38542143640", "original"),
    "c07_dense_pre_123": (C7 / "replication_38544932532", "38544932532", "original"),
    "c07_dense_rand_123": (C7 / "replication_38544932532", "38544932532", "original"),
    "c09_natgeom_pre_42": (ROOT / "screen_38578925047", "38578925047", "natgeom"),
    "c09_natgeom_rand_42": (ROOT / "screen_38578925047", "38578925047", "natgeom"),
    "c09_natgeom_pre_123": (ROOT / "replication_38583133395", "38583133395", "natgeom"),
    "c09_natgeom_rand_123": (ROOT / "replication_38583133395", "38583133395", "natgeom"),
    "c09_natgeom_pre_42_96": (ROOT / "budget96_38583138287", "38583138287", "natgeom"),
    "c09_natgeom_rand_42_96": (ROOT / "budget96_38583138287", "38583138287", "natgeom"),
}
PAIRS = [
    ("original", 32, "42", "c07_dense_pre_42", "c07_dense_rand_42"),
    ("original", 32, "123", "c07_dense_pre_123", "c07_dense_rand_123"),
    ("natgeom", 32, "42", "c09_natgeom_pre_42", "c09_natgeom_rand_42"),
    ("natgeom", 32, "123", "c09_natgeom_pre_123", "c09_natgeom_rand_123"),
    ("natgeom", 96, "42", "c09_natgeom_pre_42_96", "c09_natgeom_rand_42_96"),
]
BUNDLES = {
    "38578925047": "af2ab30053afdee3231c86dd0f7cc3709e2e859a4dd4e23fe06fdbec21dd778c",
    "38583133395": "7547f29053f0a4a2a3e2580896288749ca491e65fc94ce0902a7258801f572d1",
    "38583138287": "af105c7d6e478d44b1e0bd9b71e541f2692fe133977c9238e963cd58154120e6",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def main():
    runs, manifest = {}, {}
    for exp_id, (folder, job, condition) in SOURCES.items():
        record = json.loads((folder / "runs" / exp_id / "results.json").read_text())
        cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][exp_id]
        runs[exp_id] = {
            "job": job,
            "condition": condition,
            "macro_dev_nll": macro(record["validation"]),
            "language_dev_nll": cand["language_dev_nll"],
            "math_dev_nll": cand["math_dev_nll"],
            "language_ratio_vs_host": cand["language_ratio_vs_host"],
            "bypass_macro": cand["bypass_macro"],
            "zero_macro": cand["zero_macro"],
            "graft_relative_update": cand["graft_relative_update"],
        }
        manifest[exp_id] = {
            "job": job,
            "status": "COMPLETED",
            "record": str(folder / "runs" / exp_id / "results.json"),
            "record_sha256": hashlib.sha256(
                (folder / "runs" / exp_id / "results.json").read_bytes()
            ).hexdigest(),
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
        "campaign": 9,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "hypothesis": "H9 donor-native representation geometry (whiten -> donor-native color)",
        "host_language_dev_nll": 4.559452972412109,
        "runs": runs,
        "comparisons": comparisons,
        "promotion": False,
        "outcome": {
            "classification": "B",
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": (
                "Donor-specific advantage replicated: benefit +9.49% (seed 42), +10.37% "
                "(seed 123), +11.49% (96 steps) under identical geometry treatment; "
                "donor causally necessary (bypass/zero worse). Usefulness gates fail "
                "everywhere (language 1.80-1.96x vs 1.50 ceiling; macro 7.84-9.02 vs "
                "6.00), so no promotion and no useful inheritance claim."
            ),
        },
        "interpretation_limit": (
            "Under donor-native coloring the random sector becomes near-identity "
            "(graft relative update ~0.001; intact ~= bypass), so the twin is close "
            "to a bypass control; the pretrained child's absolute macro (8.65-9.02) "
            "is not better than under the original interface (8.57-9.40). The "
            "donor-specific effect is real against the preregistered matched "
            "control, but geometry mainly disables the twin rather than raising "
            "the pretrained child in absolute terms."
        ),
        "evidence_bundles_sha256": BUNDLES,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "ablation_results.json").write_text(
        json.dumps(
            {
                k: {
                    f: r[f]
                    for f in (
                        "macro_dev_nll",
                        "bypass_macro",
                        "zero_macro",
                        "graft_relative_update",
                    )
                }
                for k, r in runs.items()
            },
            indent=2,
        )
        + "\n"
    )
    with open(ROOT / "comparison_table.csv", "w", newline="") as handle:
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
    lines = []
    for path in sorted(ROOT.rglob("*.json")) + sorted(ROOT.rglob("*.csv")):
        if "screen_" in str(path) or "replication_" in str(path) or "budget96_" in str(path):
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    (ROOT / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print("Campaign 9 results published; outcome:", results["outcome"]["classification"])


if __name__ == "__main__":
    main()
