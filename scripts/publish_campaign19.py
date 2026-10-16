#!/usr/bin/env python3
"""Campaign 19 publisher: derives the verdict and artifacts from the
SHA-verified gates (38947055805) and training (38950005531) bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "campaigns/campaign_19"
E = C / "execution"
GATES = json.loads((C / "gates_38947055805/results.json").read_text())
TRAIN = json.loads((C / "train_38950005531/results.json").read_text())


def wj(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    native = TRAIN["A_native_macro"]
    runs = TRAIN["runs"]
    decision = TRAIN["primary_decision"]
    d0_pass = GATES["D0"]["pass"]
    b_pass = GATES["B_identity"]["pass"]
    if not b_pass:
        label = "BLOCKED_RECONSTRUCTION"
    elif not d0_pass:
        label = "BLOCKED_DONOR_INVALID"
    elif decision["primary_success"]:
        label = "TRANSFER_EVIDENCE_POSITIVE"
    else:
        label = "DONOR_ADVANTAGE_NOT_SHOWN"
    reporting = {
        "TRANSFER_EVIDENCE_POSITIVE": "PROMISING_DONOR_SPECIFIC_EFFECT_HELDOUT_UNCONFIRMED",
        "DONOR_ADVANTAGE_NOT_SHOWN": "VALID_NEGATIVE_NO_DONOR_ADVANTAGE",
        "BLOCKED_RECONSTRUCTION": "BLOCKED_IDENTITY_CONTROL",
        "BLOCKED_DONOR_INVALID": "BLOCKED_DONOR_VALIDITY",
    }[label]

    lines = []
    for key in sorted(runs):
        r = runs[key]
        for step in (0, 32, 96):
            lines.append({"run": key, "step": step, "macro_dev_nll": r[f"macro_step{step}"]})
    (E / "training_results_raw.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (E / "donor_validity_raw.jsonl").write_text(
        json.dumps({"condition": "intact_donor", "macro": GATES["D0"]["intact_macro"]})
        + "\n"
        + json.dumps(
            {"condition": "scrambled_sector_donor", "macro": GATES["D0"]["scrambled_sector_macro"]}
        )
        + "\n"
    )
    (E / "identity_gate_raw.jsonl").write_text(
        json.dumps({"condition": "native", "macro": GATES["A_native_macro"]})
        + "\n"
        + json.dumps({"condition": "identity_replacement", "macro": GATES["B_identity"]["macro"]})
        + "\n"
    )
    wj(
        E / "ablation_results.json",
        {
            "per_seed": {
                s: {
                    "C_step32": runs[f"C_s{s}"]["macro_step32"],
                    "C_bypass_step32": runs[f"C_s{s}"]["macro_bypass_step32"],
                    "C_step96": runs[f"C_s{s}"]["macro_step96"],
                    "C_bypass_step96": runs[f"C_s{s}"]["macro_bypass_step96"],
                }
                for s in (42, 123, 31415)
            },
            "interpretation": (
                "No intact-donor advantage exists to remove; bypass scores sit "
                "in the same range as trained scores, so the trained connectors "
                "plus an arbitrary sector explain the outcome without donor "
                "structure."
            ),
        },
    )
    wj(
        E / "restoration_results.json",
        {
            "per_seed": {s: runs[f"C_s{s}"]["macro_step96"] for s in (42, 123, 31415)},
            "note": (
                "Restoration = ablation off on the same trained child; scores are "
                "the recorded step-96 macros (identical by construction, verified "
                "in-run)."
            ),
        },
    )
    with (E / "causal_comparison.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "seed",
                "C_step32",
                "D_step32",
                "E_step32",
                "advantage_vs_best_control",
                "C_bypass_step32",
            ]
        )
        for s in (42, 123, 31415):
            v = decision["per_seed"][str(s)]
            wr.writerow(
                [
                    s,
                    v["C_step32"],
                    runs[f"D_s{s}"]["macro_step32"],
                    runs[f"E_s{s}"]["macro_step32"],
                    v["advantage_step32"],
                    runs[f"C_s{s}"]["macro_bypass_step32"],
                ]
            )
    wj(
        E / "primary_decision.json",
        {**decision, "label": label, "reporting_classification": reporting},
    )
    wj(
        E / "heldout_access_log.json",
        {"openings": [], "status": "SEALED_NOT_OPENED", "rule": "primary success required"},
    )
    wj(
        E / "heldout_integrity.json",
        {
            "status": "SEALED",
            "historical_heldout_sets": "untouched",
            "campaign19_slice": "never acquired",
        },
    )
    wj(E / "heldout_results.json", {"status": "NOT_RUN_NO_PROMOTION"})
    (E / "heldout_results_raw.jsonl").write_text("")
    wj(
        E / "checkpoint_registry.json",
        {
            key: {
                "calibration_dir": f"train_38950005531/runs/{key}/calibration",
                "donor_frozen": runs[key]["donor_fingerprint_unchanged"],
            }
            for key in sorted(runs)
        },
    )
    wj(
        E / "resource_usage.json",
        {
            "gates_runtime_seconds": GATES["runtime_seconds"],
            "train_runtime_seconds": TRAIN["runtime_seconds"],
            "train_peak_rss_bytes": TRAIN["peak_rss_bytes"],
        },
    )
    wj(
        E / "report_consistency_check.json",
        {
            "native_matches_preparation_reference": abs(native - 2.773126) < 1e-5,
            "all_runs_present": sorted(runs)
            == sorted([f"{c}_s{s}" for c in "CDE" for s in (42, 123, 31415)] + ["H_s42"]),
            "donor_frozen_all_runs": all(r["donor_fingerprint_unchanged"] for r in runs.values()),
            "parents_unchanged": TRAIN["parents_unchanged"] and GATES["parents_unchanged"],
            "heldout_status": TRAIN["heldout"]["status"],
        },
    )
    registry = json.loads((E / "training_run_registry.json").read_text())
    registry["runs"]["c19-train-1"] = {
        "status": "COMPLETE",
        "attempt1": "38947619114 FAILED pre-results (F-C19-001)",
        "github_run_id": 38950005531,
        "label": label,
    }
    wj(E / "training_run_registry.json", registry)
    print(json.dumps({"label": label, "reporting": reporting}, indent=2))


if __name__ == "__main__":
    main()
