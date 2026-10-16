#!/usr/bin/env python3
"""Build Campaign 13 published results from imported raw run records.

Campaign 13 (same-family positive control) compares an intact
SmolLM-135M-Instruct sector transplanted into SmolLM-135M against
within-row shuffled (B), full-tensor shuffled (C), and random-twin
(D) sectors at 32 steps (seeds 42/123/31415) and 96 steps (42/123).
The validity list, gates, arrangement-sensitivity diagnostic, and
outcome mapping are exactly those frozen in campaigns/campaign_13/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path("campaigns/campaign_13")
SCREEN_JOB = "38767422815"
BUDGET_JOB = "38770695036"
SELF_JOB = "38770694519"
CONDITIONS = {"pre": "A", "row": "B", "full": "C", "rand": "D"}
PRIMARY_SEEDS = (42, 123, 31415)
BUDGET_SEEDS = (42, 123)
BUNDLES = {
    SCREEN_JOB: "914bd262953f840fb5cd5d38c0cf0396303a028ddfc869ca4c99a78afc7be885",
    BUDGET_JOB: "bf953eb2e97efcb08d56de5931664630091c7f9cf8a63d8e4ed4d61153d4a3f8",
    SELF_JOB: "13e7c57e3daf19747590e37a6bce53649bd24771055820ade962bb5221632114",
}


def benefit_percent(control_nll: float, intact_nll: float) -> float:
    """Frozen benefit: 100 * (L_control - L_A) / L_control. Positive favors A."""
    return 100.0 * (control_nll - intact_nll) / control_nll


def ablation_criterion(intact: float, bypass: float, zero: float) -> bool:
    """Frozen validity criterion (4), preregistration wording verbatim:
    "bypass and zero ablation macros finite and above intact"."""
    return (
        math.isfinite(intact)
        and math.isfinite(bypass)
        and math.isfinite(zero)
        and bypass > intact
        and zero > intact
    )


def learned_structure_gate(primary_rows) -> bool:
    """A must beat B and C by >=2% at seed 42 and >=1% at seeds 123/31415."""
    for row in primary_rows:
        threshold = 2.0 if row["seed"] == 42 else 1.0
        if row["vs_B_percent"] < threshold or row["vs_C_percent"] < threshold:
            return False
    return True


def arrangement_sensitivity(primary_rows) -> str:
    """Frozen descriptive diagnostic: adverse detection vs none."""
    pooled = [r["vs_B_percent"] for r in primary_rows] + [r["vs_C_percent"] for r in primary_rows]
    if all(r["vs_B_percent"] < 0 and r["vs_C_percent"] < 0 for r in primary_rows) and (
        sum(pooled) / len(pooled) <= -2.0
    ):
        return "adverse_detected"
    return "none"


def classify(control_valid: bool, gate_pass: bool, causal_pass: bool, usefulness_pass: bool) -> str:
    if not control_valid:
        return "D"
    if gate_pass and causal_pass and usefulness_pass:
        return "A"
    if gate_pass and causal_pass:
        return "B"
    return "C"


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def exp_id(cond: str, seed: int, steps: int) -> str:
    return f"c13_{cond}_{seed}" + ("_96" if steps == 96 else "")


def load_run(folder: Path, job: str, eid: str, letter: str, steps: int, seed: int):
    record = json.loads((folder / "runs" / eid / "results.json").read_text())
    cand = json.loads((folder / "phase_summary.json").read_text())["candidates"][eid]
    run = {
        "job": job,
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
        "positive_control": record["structure"].get("positive_control", False),
    }
    manifest = {
        "job": job,
        "status": "COMPLETED",
        "condition": letter,
        "record_sha256": hashlib.sha256(
            (folder / "runs" / eid / "results.json").read_bytes()
        ).hexdigest(),
    }
    return run, manifest, record


def main():
    folders = {32: ROOT / f"screen32_{SCREEN_JOB}", 96: ROOT / f"budget96_{BUDGET_JOB}"}
    jobs = {32: SCREEN_JOB, 96: BUDGET_JOB}
    runs, manifest, activity = {}, {}, {}
    for steps, seeds in ((32, PRIMARY_SEEDS), (96, BUDGET_SEEDS)):
        for cond, letter in CONDITIONS.items():
            for seed in seeds:
                eid = exp_id(cond, seed, steps)
                run, man, _ = load_run(folders[steps], jobs[steps], eid, letter, steps, seed)
                runs[eid] = run
                manifest[eid] = man
                activity[eid] = {"graft_relative_update_trained": run["graft_relative_update"]}
    self_folder = ROOT / f"selftest_{SELF_JOB}"
    if self_folder.exists():
        run, man, _ = load_run(self_folder, SELF_JOB, "c13_self_42", "F", 32, 42)
        runs["c13_self_42"] = run
        manifest["c13_self_42"] = man

    native = json.loads((folders[32] / "native_development.json").read_text())
    host_language = native["recipient"]["groups"]["domain:language"]["token_nll"]
    host_macro = macro(native["recipient"])
    donor_native_macro = macro(native["donor"])

    # --- Frozen validity for B and C runs (preregistration list, verbatim) ---
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
                    "ablations_finite_and_above_intact": ablation_criterion(
                        r["macro_dev_nll"], r["bypass_macro"], r["zero_macro"]
                    ),
                    # Descriptive only, not a validity gate.
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
            row["A_causal_bypass_above_intact"] = a["bypass_macro"] > a["macro_dev_nll"]
            comparisons.append(row)
    primary = [c for c in comparisons if c["steps"] == 32]
    gate_pass = learned_structure_gate(primary)
    causal_pass = all(c["A_causal_bypass_above_intact"] for c in primary)
    usefulness_pass = all(c["A_G_C_pass"] for c in primary)
    sensitivity = arrangement_sensitivity(primary)
    classification = classify(control_valid, gate_pass, causal_pass, usefulness_pass)
    positive_control_pass = bool(control_valid and gate_pass and causal_pass)
    rationale = {
        "A": "Positive control passes and usefulness gates pass.",
        "B": (
            "Positive control passes (pipeline preserves learned structure "
            "same-family); usefulness fails."
        ),
        "C": (
            "Controls valid; intact sector shows no reproducible advantage "
            "over its scrambled selves."
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
        "campaign": 13,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "type": "same_family_positive_control",
        "hypothesis": (
            "H13: pipeline preserves learned structure in a compatible same-family setting"
        ),
        "host_language_dev_nll": host_language,
        "host_macro_dev_nll": host_macro,
        "donor_native_macro_dev_nll": donor_native_macro,
        "shuffle_seed": 20241014,
        "runs": runs,
        "comparisons": comparisons,
        "descriptive_primary": descriptive,
        "control_valid": control_valid,
        "learned_structure_gate_pass": gate_pass,
        "causal_support": causal_pass,
        "positive_control_pass": positive_control_pass,
        "arrangement_sensitivity": sensitivity,
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
        if any(
            tag in str(path)
            for tag in ("screen32_", "budget96_", "validate_", "probe_", "selftest_")
        ):
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    (ROOT / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(
        "Campaign 13 results published; outcome:",
        classification,
        "| positive_control_pass:",
        positive_control_pass,
        "| sensitivity:",
        sensitivity,
    )


if __name__ == "__main__":
    main()
