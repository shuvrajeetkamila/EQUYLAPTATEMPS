#!/usr/bin/env python3
"""Build Campaign 12 published results from imported raw run records.

Campaign 12 (The Structure-Scale Test) compares the intact donor
sector against within-row shuffled (B), full-tensor shuffled (C,
Campaign 11's control), and random-twin (D) sectors at 32 steps
(seeds 42/123/31415) and 96 steps (seeds 42/123). Gates and outcome
mapping are exactly those frozen in campaigns/campaign_12/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path("campaigns/campaign_12")
SCREEN_JOB = "38726922262"
BUDGET_JOB = "38738479882"
CONDITIONS = {"pre": "A", "row": "B", "full": "C", "rand": "D"}
PRIMARY_SEEDS = (42, 123, 31415)
BUDGET_SEEDS = (42, 123)
HOST_LANGUAGE_NLL = 4.559452953338623
BUNDLES = {
    SCREEN_JOB: "346a3f416e801007e2ff44256c931e269ac7b2e3a389cb3b81c6c3f50fe6aa59",
    BUDGET_JOB: "a46764db2a45912fe28e57c93c6471fc75f23e4231bdadac6d8ef242fdad90fe",
}


def benefit_percent(control_nll: float, intact_nll: float) -> float:
    """Frozen benefit: 100 * (L_control - L_A) / L_control. Positive favors A."""
    return 100.0 * (control_nll - intact_nll) / control_nll


def learned_structure_gate(primary_rows) -> bool:
    """A must beat B and C by >=2% at seed 42 and >=1% at seeds 123/31415."""
    for row in primary_rows:
        threshold = 2.0 if row["seed"] == 42 else 1.0
        if row["vs_B_percent"] < threshold or row["vs_C_percent"] < threshold:
            return False
    return True


def classify(control_valid: bool, gate_pass: bool, usefulness_pass: bool) -> str:
    if not control_valid:
        return "D"
    if gate_pass and usefulness_pass:
        return "A"
    if gate_pass:
        return "B"
    return "C"


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def exp_id(cond: str, seed: int, steps: int) -> str:
    return f"c12_{cond}_{seed}" + ("_96" if steps == 96 else "")


def main():
    folders = {32: ROOT / f"screen32_{SCREEN_JOB}", 96: ROOT / f"budget96_{BUDGET_JOB}"}
    jobs = {32: SCREEN_JOB, 96: BUDGET_JOB}
    runs, manifest, activity = {}, {}, {}
    for steps, seeds in ((32, PRIMARY_SEEDS), (96, BUDGET_SEEDS)):
        for cond, letter in CONDITIONS.items():
            for seed in seeds:
                eid = exp_id(cond, seed, steps)
                folder = folders[steps]
                record = json.loads((folder / "runs" / eid / "results.json").read_text())
                cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][eid]
                runs[eid] = {
                    "job": jobs[steps],
                    "condition": letter,
                    "steps": steps,
                    "seed": seed,
                    "macro_dev_nll": macro(record["validation"]),
                    "language_dev_nll": cand["language_dev_nll"],
                    "math_dev_nll": cand["math_dev_nll"],
                    "language_ratio_vs_host": cand["language_ratio_vs_host"],
                    "bypass_macro": cand["bypass_macro"],
                    "zero_macro": cand["zero_macro"],
                    "graft_relative_update": cand["graft_relative_update"],
                    "parent_weights_unchanged": record["training"]["parent_weights_unchanged"],
                }
                manifest[eid] = {
                    "job": jobs[steps],
                    "status": "COMPLETED",
                    "condition": letter,
                    "record_sha256": hashlib.sha256(
                        (folder / "runs" / eid / "results.json").read_bytes()
                    ).hexdigest(),
                }
                activity[eid] = {"graft_relative_update_trained": cand["graft_relative_update"]}

    # --- Frozen post-training control validity for B and C runs ---
    validity = {}
    for steps, seeds in ((32, PRIMARY_SEEDS), (96, BUDGET_SEEDS)):
        for cond in ("row", "full"):
            for seed in seeds:
                eid = exp_id(cond, seed, steps)
                init = json.loads(
                    (folders[steps] / "runs" / eid / "initialization.json").read_text()
                )
                control = init["sector_control"]
                r = runs[eid]
                checks = {
                    "kind": control["kind"],
                    "eligible_tensors": control["eligible_tensors"],
                    "all_multisets_preserved": bool(control["all_multisets_preserved"]),
                    "row_multisets_preserved": bool(
                        control.get("all_row_multisets_preserved", True)
                    ),
                    "all_eligible_actually_permuted": all(
                        t["sha256_before"] != t["sha256_after"]
                        for t in control["tensors"]
                        if t["eligible"]
                    ),
                    "graft_update_at_least_0_05": r["graft_relative_update"] >= 0.05,
                    # Frozen criterion (4) as written in the preregistration:
                    # "bypass and zero ablation macros finite and above intact"
                    # (see campaigns/campaign_12/amendment_01_validity_criterion.md).
                    "ablations_finite_and_above_intact": all(
                        math.isfinite(r[f]) for f in ("macro_dev_nll", "bypass_macro", "zero_macro")
                    )
                    and r["bypass_macro"] > r["macro_dev_nll"]
                    and r["zero_macro"] > r["macro_dev_nll"],
                    # Descriptive only, not a validity gate (Campaign 11-style ordering).
                    "bypass_below_zero_ordering": r["bypass_macro"] < r["zero_macro"],
                    "parent_weights_unchanged": r["parent_weights_unchanged"],
                }
                checks["valid"] = all(
                    v
                    for k, v in checks.items()
                    if k not in ("kind", "bypass_below_zero_ordering", "eligible_tensors")
                )
                validity[eid] = checks
    control_valid = all(v["valid"] for v in validity.values())

    # --- Matched comparisons ---
    comparisons = []
    for steps, seeds in ((32, PRIMARY_SEEDS), (96, BUDGET_SEEDS)):
        for seed in seeds:
            a = runs[exp_id("pre", seed, steps)]
            row = {"steps": steps, "seed": seed, "A_intact_macro": a["macro_dev_nll"]}
            for cond, key in (("row", "B"), ("full", "C"), ("rand", "D")):
                control = runs[exp_id(cond, seed, steps)]
                row[f"{key}_macro"] = control["macro_dev_nll"]
                row[f"vs_{key}_percent"] = benefit_percent(
                    control["macro_dev_nll"], a["macro_dev_nll"]
                )
            row["A_language_ratio_vs_host"] = a["language_ratio_vs_host"]
            row["A_G_C_pass"] = a["language_ratio_vs_host"] <= 1.50 and a["macro_dev_nll"] <= 6.00
            comparisons.append(row)
    primary = [c for c in comparisons if c["steps"] == 32]
    gate_pass = learned_structure_gate(primary)
    usefulness_pass = all(c["A_G_C_pass"] for c in primary)
    classification = classify(control_valid, gate_pass, usefulness_pass)
    rationale = {
        "A": "Replicated learned-structure advantage and usefulness gates pass.",
        "B": "Replicated learned-structure advantage; usefulness gates fail.",
        "C": (
            "Controls valid, but the intact donor does not beat both scrambled "
            "controls by the frozen per-seed margins at all primary seeds."
        ),
        "D": "A scrambled control failed a frozen validity criterion; inconclusive.",
    }[classification]

    descriptive = {}
    for key in ("vs_B_percent", "vs_C_percent", "vs_D_percent"):
        values = [c[key] for c in primary]
        descriptive[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "note": "descriptive only (n=3 seeds); no significance claim",
        }

    results = {
        "campaign": 12,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "hypothesis": (
            "H12: intact donor beats within-row and full-tensor scrambled selves reproducibly"
        ),
        "host_language_dev_nll": HOST_LANGUAGE_NLL,
        "shuffle_seed": 20241014,
        "runs": runs,
        "comparisons": comparisons,
        "descriptive_primary": descriptive,
        "control_valid": control_valid,
        "learned_structure_gate_pass": gate_pass,
        "promotion": False,
        "outcome": {
            "classification": classification,
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": rationale,
        },
        "evidence_bundles_sha256": BUNDLES,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    validation_path = ROOT / "control_validation.json"
    validation_doc = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    validation_doc["post_training_run_checks"] = {"control_valid": control_valid, "runs": validity}
    validation_path.write_text(json.dumps(validation_doc, indent=2) + "\n")

    ledger_path = ROOT / "run_registry.json"
    ledger = json.loads(ledger_path.read_text())
    for eid, entry in manifest.items():
        ledger["runs"][eid] = {
            **ledger["runs"].get(eid, {}),
            "status": "COMPLETED",
            "job": entry["job"],
        }
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "activity_diagnostics.json").write_text(json.dumps({"runs": activity}, indent=2) + "\n")
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
                "A_intact",
                "B_row",
                "C_full",
                "D_rand",
                "vs_B_percent",
                "vs_C_percent",
                "vs_D_percent",
                "A_G_C_pass",
            ]
        )
        for c in comparisons:
            writer.writerow(
                [
                    c["steps"],
                    c["seed"],
                    f"{c['A_intact_macro']:.4f}",
                    f"{c['B_macro']:.4f}",
                    f"{c['C_macro']:.4f}",
                    f"{c['D_macro']:.4f}",
                    f"{c['vs_B_percent']:.2f}",
                    f"{c['vs_C_percent']:.2f}",
                    f"{c['vs_D_percent']:.2f}",
                    c["A_G_C_pass"],
                ]
            )
    lines = []
    for path in sorted(ROOT.rglob("*.json")) + sorted(ROOT.rglob("*.csv")):
        if any(tag in str(path) for tag in ("screen32_", "budget96_", "validate_")):
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    (ROOT / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(
        "Campaign 12 results published; outcome:", classification, "| control_valid:", control_valid
    )


if __name__ == "__main__":
    main()
