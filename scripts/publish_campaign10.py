#!/usr/bin/env python3
"""Build Campaign 10 published results from imported raw run records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("campaigns/campaign_10")
SOURCES = {
    "c10_out_pre_42": (ROOT / "screen_38602501026", "38602501026", "B"),
    "c10_out_rand_42": (ROOT / "screen_38602501026", "38602501026", "C"),
    "c10_out_awake_42": (ROOT / "screen_38602501026", "38602501026", "D_exploratory"),
    "c10_out_awake_42_amd1": (ROOT / "awakeamd_38606201339", "38606201339", "D"),
    "c10_out_pre_123": (ROOT / "replication_38606208077", "38606208077", "B"),
    "c10_out_rand_123": (ROOT / "replication_38606208077", "38606208077", "C"),
    "c10_out_awake_123": (ROOT / "replication_38606208077", "38606208077", "D"),
    "c10_out_pre_42_96": (ROOT / "budget96_38607522188", "38607522188", "B"),
    "c10_out_rand_42_96": (ROOT / "budget96_38607522188", "38607522188", "C"),
    "c10_out_awake_42_96": (ROOT / "budget96_38607522188", "38607522188", "D"),
}
TRIPLES = [
    (32, "42", "c10_out_pre_42", "c10_out_rand_42", "c10_out_awake_42_amd1"),
    (32, "123", "c10_out_pre_123", "c10_out_rand_123", "c10_out_awake_123"),
    (96, "42", "c10_out_pre_42_96", "c10_out_rand_42_96", "c10_out_awake_42_96"),
]
BUNDLES = {
    "38602501026": "e8d1057794c449edf636f596a58efb779674f4ca30be9331ff28687c94789e01",
    "38606201339": "5e11d379aed4a1413e084d07fd0a89acf2432924c19a7fdb9f9ddc44d73bf760",
    "38606208077": "25a6d57988d07a88059915abca984ad367bc96bb64a3e99797c22103cab62795",
    "38607522188": "bf87886d6bebf8aa7979b88fec0d326f863efe66293cfafb865a036dd6bbf0f4",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def benefit(twin, pre):
    return (twin - pre) / twin


def main():
    runs, manifest, activity = {}, {}, {}
    for exp_id, (folder, job, condition) in SOURCES.items():
        record = json.loads((folder / "runs" / exp_id / "results.json").read_text())
        cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][exp_id]
        init = json.loads((folder / "runs" / exp_id / "initialization.json").read_text())
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
            "trainable_parameters": record["structure"]["trainable_dense_seam_parameters"],
        }
        manifest[exp_id] = {
            "job": job,
            "status": "COMPLETED",
            "condition": condition,
            "record_sha256": hashlib.sha256(
                (folder / "runs" / exp_id / "results.json").read_bytes()
            ).hexdigest(),
        }
        activity[exp_id] = {
            "graft_relative_update_trained": cand["graft_relative_update"],
            "awake_calibration": init.get("awake_calibration"),
        }
    target = json.loads((ROOT / "screen_38602501026" / "awake_calibration_target.json").read_text())
    comparisons = []
    for steps, seed, b_id, c_id, d_id in TRIPLES:
        b, c, d = runs[b_id], runs[c_id], runs[d_id]
        comparisons.append(
            {
                "steps": steps,
                "seed": seed,
                "B_pretrained_macro": b["macro_dev_nll"],
                "C_raw_twin_macro": c["macro_dev_nll"],
                "D_awake_twin_macro": d["macro_dev_nll"],
                "benefit_vs_D_fraction": benefit(d["macro_dev_nll"], b["macro_dev_nll"]),
                "benefit_vs_C_fraction": benefit(c["macro_dev_nll"], b["macro_dev_nll"]),
                "G_B_vs_D_pass": b["macro_dev_nll"] <= 0.98 * d["macro_dev_nll"]
                if seed == "42"
                else b["macro_dev_nll"] <= 0.99 * d["macro_dev_nll"],
                "G_C_pass": b["language_ratio_vs_host"] <= 1.50 and b["macro_dev_nll"] <= 6.00,
            }
        )
    results = {
        "campaign": 10,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "hypothesis": "H10: C9 advantage survives awake twin + trained output pathway",
        "host_language_dev_nll": 4.559452953338623,
        "awake_target_relative_update": target["mean_relative_update"],
        "runs": runs,
        "comparisons": comparisons,
        "promotion": False,
        "outcome": {
            "classification": "B",
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": (
                "Frozen G-B vs the awake twin passed at both 32-step seeds (+10.43%, "
                "+4.42%) but usefulness G-C failed everywhere (best macro 7.8392 vs "
                "6.00; best language 1.835x vs 1.50). Advantage is fragile: vs the "
                "raw twin it is +5.03%/+0.97% at 32 steps and -1.69% at 96 steps."
            ),
        },
        "evidence_bundles_sha256": BUNDLES,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "run_registry.json").write_text(json.dumps({"runs": manifest}, indent=2) + "\n")
    (ROOT / "activity_diagnostics.json").write_text(
        json.dumps({"awake_target": target, "runs": activity}, indent=2) + "\n"
    )
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
                "steps",
                "seed",
                "B_pretrained",
                "C_raw_twin",
                "D_awake_twin",
                "benefit_vs_D_percent",
                "benefit_vs_C_percent",
                "G_B_vs_D_pass",
                "G_C_pass",
            ]
        )
        for c in comparisons:
            writer.writerow(
                [
                    c["steps"],
                    c["seed"],
                    f"{c['B_pretrained_macro']:.4f}",
                    f"{c['C_raw_twin_macro']:.4f}",
                    f"{c['D_awake_twin_macro']:.4f}",
                    f"{100 * c['benefit_vs_D_fraction']:.2f}",
                    f"{100 * c['benefit_vs_C_fraction']:.2f}",
                    c["G_B_vs_D_pass"],
                    c["G_C_pass"],
                ]
            )
    lines = []
    for path in sorted(ROOT.rglob("*.json")) + sorted(ROOT.rglob("*.csv")):
        if any(tag in str(path) for tag in ("screen_", "replication_", "budget96_", "awakeamd_")):
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    (ROOT / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print("Campaign 10 results published; outcome:", results["outcome"]["classification"])


if __name__ == "__main__":
    main()
