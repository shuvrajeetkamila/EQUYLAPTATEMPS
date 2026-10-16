#!/usr/bin/env python3
"""Build Campaign 7 published results + CSVs from imported raw run records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("campaigns/campaign_07")
JOBS = {
    "c07_dense_pre_42": ("screen_38542143640", "38542143640"),
    "c07_dense_rand_42": ("screen_38542143640", "38542143640"),
    "c07_dense_pre_123": ("replication_38544932532", "38544932532"),
    "c07_dense_rand_123": ("replication_38544932532", "38544932532"),
}
BUNDLES = {
    "38542143640": "d568e599954074b274ad4b69330d69977c5b034e7f9ef0e7c3ca1bbad0be6792",
    "38544932532": "3c59160e6d561399478d636a978b35b7eb66035549e050be0445cd87492236e2",
}


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def main():
    runs = {}
    for exp_id, (folder, job) in JOBS.items():
        record = json.loads((ROOT / folder / "runs" / exp_id / "results.json").read_text())
        summary = json.loads((ROOT / folder / "phase_summary.json").read_text())
        cand = summary["candidates"][exp_id]
        runs[exp_id] = {
            "job": job,
            "status": "COMPLETED",
            "macro_dev_nll": macro(record["validation"]),
            "language_dev_nll": cand["language_dev_nll"],
            "math_dev_nll": cand["math_dev_nll"],
            "language_ratio_vs_host": cand["language_ratio_vs_host"],
            "bypass_macro": cand["bypass_macro"],
            "zero_macro": cand["zero_macro"],
            "graft_relative_update": cand["graft_relative_update"],
            "structure": record["structure"],
        }
    host = json.loads((ROOT / "screen_38542143640" / "phase_summary.json").read_text())[
        "host_language_dev_nll"
    ]
    gates = {}
    for seed in (42, 123):
        pre, twin = runs[f"c07_dense_pre_{seed}"], runs[f"c07_dense_rand_{seed}"]
        benefit = (twin["macro_dev_nll"] - pre["macro_dev_nll"]) / twin["macro_dev_nll"]
        gates[str(seed)] = {
            "pretrained_macro": pre["macro_dev_nll"],
            "twin_macro": twin["macro_dev_nll"],
            "benefit_fraction": benefit,
            "G_B_pass": pre["macro_dev_nll"] <= 0.98 * twin["macro_dev_nll"],
            "G_C_pass": pre["language_ratio_vs_host"] <= 1.50 and pre["macro_dev_nll"] <= 6.00,
        }
    promoted = all(g["G_B_pass"] and g["G_C_pass"] for g in gates.values())
    results = {
        "campaign": 7,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "pairing": {
            "recipient": "state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
            "donor": "HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4",
            "donor_sector": [24, 30],
            "recipient_cut": [11, 13],
            "interface": "dense full-size layernorm_linear seams (768<->576)",
            "trainable_parameters": runs["c07_dense_pre_42"]["structure"][
                "trainable_dense_seam_parameters"
            ],
        },
        "host_language_dev_nll": host,
        "runs": runs,
        "gates": gates,
        "promotion": promoted,
        "outcome": {
            "classification": "C",
            "held_out_status": "NOT_RUN_NO_PROMOTION",
            "rationale": (
                "Valid two-seed experiment; the matched random twin beat the pretrained "
                "SmolLM sector at both seeds (benefit -13.8% and -19.2%), so G-B fails "
                "and no reliable pretrained advantage exists for this pairing/design."
            ),
        },
        "evidence": {
            "screen_job": "38542143640",
            "screen_bundle_sha256": BUNDLES["38542143640"],
            "replication_job": "38544932532",
            "replication_bundle_sha256": BUNDLES["38544932532"],
        },
    }
    (ROOT / "campaign_07_results.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(ROOT / "control_comparisons.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "pretrained_macro", "twin_macro", "benefit_percent", "G_B_pass", "G_C_pass"]
        )
        for seed in ("42", "123"):
            g = gates[seed]
            writer.writerow(
                [
                    seed,
                    f"{g['pretrained_macro']:.4f}",
                    f"{g['twin_macro']:.4f}",
                    f"{100 * g['benefit_fraction']:.2f}",
                    g["G_B_pass"],
                    g["G_C_pass"],
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
    print("Campaign 7 results published; outcome:", results["outcome"]["classification"])


if __name__ == "__main__":
    main()
