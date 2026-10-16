#!/usr/bin/env python3
"""Build Campaign 11 published results from imported raw run records.

Campaign 11 (The Learned-Structure Test) asks whether the intact
pretrained donor sector beats a statistics-preserved,
structure-destroyed (within-tensor permuted) version of itself under
identical treatment. Condition A is read from the stored Campaign 10
condition-B records; condition B from the Campaign 11 runs. The
control validity criteria, benefit definition, gates, and outcome
mapping are exactly those frozen in campaigns/campaign_11/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT11 = Path("campaigns/campaign_11")
ROOT10 = Path("campaigns/campaign_10")
BUDGET_JOB = "38715108842"  # filled from the imported budget-96 bundle

# exp_id -> (evidence folder, actions job, condition)
SOURCES = {
    "c10_out_pre_42": (ROOT10 / "screen_38602501026", "38602501026", "A"),
    "c10_out_pre_123": (ROOT10 / "replication_38606208077", "38606208077", "A"),
    "c10_out_pre_42_96": (ROOT10 / "budget96_38607522188", "38607522188", "A"),
    "c10_out_rand_42": (ROOT10 / "screen_38602501026", "38602501026", "C_reference"),
    "c10_out_rand_123": (ROOT10 / "replication_38606208077", "38606208077", "C_reference"),
    "c11_shuf_42": (ROOT11 / "screen_38713676113", "38713676113", "B"),
    "c11_shuf_123": (ROOT11 / "screen_38713676113", "38713676113", "B"),
    "c11_shuf_42_96": (ROOT11 / f"budget96_{BUDGET_JOB}", BUDGET_JOB, "B"),
}
PAIRS = [  # (steps, seed, A exp, B exp) — primary first two, budget last
    (32, "42", "c10_out_pre_42", "c11_shuf_42"),
    (32, "123", "c10_out_pre_123", "c11_shuf_123"),
    (96, "42", "c10_out_pre_42_96", "c11_shuf_42_96"),
]
BUNDLES = {
    "38602501026": "e8d1057794c449edf636f596a58efb779674f4ca30be9331ff28687c94789e01",
    "38606208077": "25a6d57988d07a88059915abca984ad367bc96bb64a3e99797c22103cab62795",
    "38607522188": "bf87886d6bebf8aa7979b88fec0d326f863efe66293cfafb865a036dd6bbf0f4",
    "38713676113": "5876079a9ad7e0eaef361f5e4e8d3e8eb9a0e860f1c96d305479bb54d2d772c2",
    BUDGET_JOB: "b8db0ce993e3bcf35c0af9a5b650dac0ce7599e4c504d6b20a454735f63e3c13",
}
HOST_LANGUAGE_NLL = 4.559452953338623


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def load_run(exp_id):
    folder, job, condition = SOURCES[exp_id]
    record = json.loads((folder / "runs" / exp_id / "results.json").read_text())
    cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][exp_id]
    init = json.loads((folder / "runs" / exp_id / "initialization.json").read_text())
    return folder, job, condition, record, cand, init


def main():
    runs, manifest, activity = {}, {}, {}
    for exp_id in SOURCES:
        folder, job, condition, record, cand, init = load_run(exp_id)
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
            "parent_weights_unchanged": record["training"]["parent_weights_unchanged"],
        }
        manifest[exp_id] = {
            "job": job,
            "status": "COMPLETED",
            "condition": condition,
            "record_sha256": hashlib.sha256(
                (folder / "runs" / exp_id / "results.json").read_bytes()
            ).hexdigest(),
        }
        activity[exp_id] = {"graft_relative_update_trained": cand["graft_relative_update"]}

    # --- Frozen control-validity criteria, checked per condition-B run ---
    validation = {}
    for exp_id in ("c11_shuf_42", "c11_shuf_123", "c11_shuf_42_96"):
        folder, job, condition, record, cand, init = load_run(exp_id)
        control = init["sector_control"]
        eligible = [t for t in control["tensors"] if t["eligible"]]
        intact = runs[exp_id]["macro_dev_nll"]
        ordered = (
            math.isfinite(intact)
            and intact < runs[exp_id]["bypass_macro"] < runs[exp_id]["zero_macro"]
        )
        checks = {
            "eligible_tensors": len(eligible),
            "all_multisets_preserved": bool(control["all_multisets_preserved"])
            and all(t["multiset_preserved"] for t in eligible),
            "all_eligible_actually_permuted": all(
                t["sha256_before"] != t["sha256_after"] for t in eligible
            ),
            "shuffle_seed": control["shuffle_seed"],
            "graft_relative_update": runs[exp_id]["graft_relative_update"],
            "graft_update_at_least_0_05": runs[exp_id]["graft_relative_update"] >= 0.05,
            "ablations_finite_and_ordered": ordered,
            "parent_weights_unchanged": runs[exp_id]["parent_weights_unchanged"],
        }
        checks["valid"] = all(
            [
                checks["all_multisets_preserved"],
                checks["all_eligible_actually_permuted"],
                checks["graft_update_at_least_0_05"],
                checks["ablations_finite_and_ordered"],
                checks["parent_weights_unchanged"],
            ]
        )
        validation[exp_id] = checks
    control_valid = all(v["valid"] for v in validation.values())

    # --- Frozen comparisons: benefit = 100 * (L_B - L_A) / L_B ---
    comparisons = []
    for steps, seed, a_id, b_id in PAIRS:
        a, b = runs[a_id], runs[b_id]
        comparisons.append(
            {
                "steps": steps,
                "seed": seed,
                "A_intact_macro": a["macro_dev_nll"],
                "B_shuffled_macro": b["macro_dev_nll"],
                "benefit_percent": 100
                * (b["macro_dev_nll"] - a["macro_dev_nll"])
                / b["macro_dev_nll"],
                "gate_seed42_min_2pct": None,
                "gate_seed123_min_1pct": None,
                "A_language_ratio_vs_host": a["language_ratio_vs_host"],
                "A_G_C_pass": a["language_ratio_vs_host"] <= 1.50 and a["macro_dev_nll"] <= 6.00,
            }
        )
    primary = comparisons[:2]
    primary[0]["gate_seed42_min_2pct"] = primary[0]["benefit_percent"] >= 2.0
    primary[1]["gate_seed123_min_1pct"] = primary[1]["benefit_percent"] >= 1.0
    primary_pass = bool(primary[0]["gate_seed42_min_2pct"] and primary[1]["gate_seed123_min_1pct"])
    usefulness_pass = all(c["A_G_C_pass"] for c in primary)

    if not control_valid:
        classification, rationale = (
            "D",
            (
                "At least one frozen control-validity criterion failed; the "
                "learned-structure question is inconclusive in this campaign."
            ),
        )
    elif primary_pass and usefulness_pass:
        classification, rationale = (
            "A",
            ("Replicated learned-structure advantage and all usefulness gates pass."),
        )
    elif primary_pass:
        classification, rationale = (
            "B",
            (
                "Replicated learned-structure advantage, but usefulness gates fail "
                "(language ratio above 1.50x host and/or macro above 6.00)."
            ),
        )
    else:
        classification, rationale = (
            "C",
            (
                "Valid statistics-preserved control, but no reliable "
                "learned-structure advantage: the intact sector's benefit over "
                "its permuted twin did not meet the frozen per-seed thresholds "
                "at both 32-step seeds."
            ),
        )

    results = {
        "campaign": 11,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "hypothesis": (
            "H11: intact pretrained sector outperforms its statistics-preserved, "
            "structure-destroyed (within-tensor permuted) twin under identical treatment"
        ),
        "host_language_dev_nll": HOST_LANGUAGE_NLL,
        "shuffle_seed": 20241014,
        "runs": runs,
        "comparisons": comparisons,
        "control_valid": control_valid,
        "primary_gate_pass": primary_pass,
        "promotion": False,
        "outcome": {
            "classification": classification,
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": rationale,
        },
        "evidence_bundles_sha256": BUNDLES,
    }
    (ROOT11 / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (ROOT11 / "control_validation.json").write_text(
        json.dumps({"control_valid": control_valid, "runs": validation}, indent=2) + "\n"
    )
    (ROOT11 / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT11 / "run_registry.json").write_text(json.dumps({"runs": manifest}, indent=2) + "\n")
    (ROOT11 / "activity_diagnostics.json").write_text(
        json.dumps({"runs": activity}, indent=2) + "\n"
    )
    (ROOT11 / "ablation_results.json").write_text(
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
    with open(ROOT11 / "comparison_table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "steps",
                "seed",
                "A_intact",
                "B_shuffled",
                "benefit_percent",
                "primary_gate",
                "A_G_C_pass",
            ]
        )
        for c, gate in zip(
            comparisons,
            [
                comparisons[0]["gate_seed42_min_2pct"],
                comparisons[1]["gate_seed123_min_1pct"],
                "secondary",
            ],
            strict=True,
        ):
            writer.writerow(
                [
                    c["steps"],
                    c["seed"],
                    f"{c['A_intact_macro']:.4f}",
                    f"{c['B_shuffled_macro']:.4f}",
                    f"{c['benefit_percent']:.2f}",
                    gate,
                    c["A_G_C_pass"],
                ]
            )
    lines = []
    for path in sorted(ROOT11.rglob("*.json")) + sorted(ROOT11.rglob("*.csv")):
        if any(tag in str(path) for tag in ("screen_", "budget96_")):
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    (ROOT11 / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(
        "Campaign 11 results published; outcome:",
        classification,
        "| control_valid:",
        control_valid,
    )


if __name__ == "__main__":
    main()
