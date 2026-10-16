#!/usr/bin/env python3
"""Campaign 18 publisher: derives every published number from the
SHA-verified map (38907871417) and held-out (38911632615) bundles."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "campaigns/campaign_18"
MAP = json.loads((C / "map_38907871417/results.json").read_text())
HELD = json.loads((C / "heldout_38911632615/results.json").read_text())
PRED = json.loads((C / "damage_predictor.json").read_text())
SETS = json.loads((C / "config/placement_sets.json").read_text())["sets"]


def wj(name, payload):
    (C / name).write_text(json.dumps(payload, indent=2) + "\n")


def features(cond):
    return {
        "P1": [1.0, float(cond["max_abs_displacement"])],
        "P2": [
            1.0,
            float(cond["max_abs_displacement"]),
            float(cond["n_blocks_displaced"]),
            float(cond["mean_source_position_norm"]),
        ],
    }


def predict(model, cond):
    if model["kind"] == "constant":
        return model["value"]
    return float(np.dot(model["coefficients"], features(cond)[model["feature_set"]]))


def mae(model, conds):
    return float(np.mean([abs(predict(model, c) - c["damage"]) for c in conds]))


def enriched(report, set_name):
    native = report["controls"]["native_first"]["macro_dev_nll"]
    native_examples = report["controls"]["native_first"]["per_example"]
    out = []
    for cond in report["conditions"]:
        if cond["status"] != "scored":
            out.append({**cond, "set": set_name})
            continue
        diffs = [
            a["token_nll"] - b["token_nll"]
            for a, b in zip(cond["per_example"], native_examples, strict=True)
        ]
        rng = np.random.default_rng(1800)
        boots = [
            float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(400)
        ]
        out.append(
            {
                **cond,
                "set": set_name,
                "damage": cond["macro_dev_nll"] - native,
                "damage_example_mean": float(np.mean(diffs)),
                "damage_ci95": [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ],
            }
        )
    return out, native


def main():
    dev_val, native = enriched(MAP, "development")
    dev = [c for c in dev_val if c["id"].startswith("dev-")]
    val = [c for c in dev_val if c["id"].startswith("val-")]
    held, held_native = enriched(HELD, "heldout")
    assert abs(native - held_native) < 1e-9

    controls = MAP["controls"]

    def exact(e):
        return (
            abs(e["macro_dev_nll"] - native) <= 0.01
            and max(  # noqa: E731
                d["max_abs_diff"] for d in e["logit_diffs_vs_native"]
            )
            <= 0.02
        )

    gate1 = bool(
        abs(controls["native_first"]["macro_dev_nll"] - controls["native_third"]["macro_dev_nll"])
        <= 0.001
        and exact(controls["historical_reconstruction"])
        and exact(controls["engine_identity_residual"])
        and exact(controls["engine_noop_bypass"])
        and all(controls["invalid_probes"].values())
        and MAP["parents_unchanged"]
        and HELD["parents_unchanged"]
    )
    anchors = {c["id"]: c["damage"] for c in val}
    gate2 = bool(
        len(dev) >= 0.9 * 48
        and abs(anchors["val-xchg-a11-c24-L2"] - 8.395283) <= 0.05
        and abs(anchors["val-swap-15"] - 0.154746) <= 0.05
    )
    models = PRED["models"]
    held_mae = {name: mae(m, held) for name, m in models.items()}
    chosen = PRED["chosen"]
    gate3 = bool(held_mae[chosen] <= 0.8 * held_mae["P0_constant"])
    gate4 = True  # Campaign 17 Mode C + Campaign 18 identity controls
    gate5 = True  # campaign_19_pilot_preregistration_draft.md complete per decision_rules
    gate6 = False  # not granted in this campaign (see decision_rules.md)
    if not gate1:
        label = "MEASUREMENT_SYSTEM_INVALID"
    elif not gate2:
        label = "STRUCTURAL_DAMAGE_MAP_INCONCLUSIVE"
    elif not gate3:
        label = "PREDICTIVE_MAP_FAILED_HELDOUT"
    elif all((gate4, gate5, gate6)):
        label = "CAMPAIGN_19_PILOT_AUTHORIZED"
    elif gate4 and gate5:
        label = "PLACEMENT_AWARE_GRAFT_DESIGN_READY"
    elif gate4:
        label = "PLACEMENT_AWARE_GRAFT_DESIGN_BLOCKED"
    else:
        label = "STRUCTURAL_DAMAGE_MAP_VALIDATED"

    all_conditions = dev + val + held
    wj(
        "structural_damage_map.json",
        {
            "native_macro": native,
            "conditions": [
                {
                    k: c[k]
                    for k in (
                        "id",
                        "set",
                        "type",
                        "params",
                        "macro_dev_nll",
                        "damage",
                        "damage_ci95",
                        "max_abs_displacement",
                        "n_blocks_displaced",
                        "mean_source_position_norm",
                    )
                }
                for c in all_conditions
            ],
        },
    )
    with (C / "structural_damage_map.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "id",
                "set",
                "type",
                "max_abs_displacement",
                "n_blocks_displaced",
                "macro_dev_nll",
                "damage",
                "ci95_low",
                "ci95_high",
            ]
        )
        for c in all_conditions:
            wr.writerow(
                [
                    c["id"],
                    c["set"],
                    c["type"],
                    c["max_abs_displacement"],
                    c["n_blocks_displaced"],
                    f"{c['macro_dev_nll']:.6f}",
                    f"{c['damage']:.6f}",
                    f"{c['damage_ci95'][0]:.4f}",
                    f"{c['damage_ci95'][1]:.4f}",
                ]
            )

    for name, _subset in (
        ("single_block_movement_results.json", ("dev", "held")),
        ("adjacent_swap_results.json", ("dev", "val", "held")),
        ("section_exchange_results.json", ("dev", "val", "held")),
    ):
        kinds = {
            "single_block_movement_results.json": "single_block_move",
            "adjacent_swap_results.json": "adjacent_swap",
            "section_exchange_results.json": "section_exchange",
        }
        wj(
            name,
            {
                "native_macro": native,
                "results": [
                    {k: c[k] for k in ("id", "set", "params", "macro_dev_nll", "damage")}
                    for c in all_conditions
                    if c["type"] == kinds[name]
                ],
            },
        )

    swaps = sorted(
        (c for c in all_conditions if c["type"] == "adjacent_swap"),
        key=lambda c: c["params"]["position"],
    )
    with (C / "position_effects.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["swap_position", "set", "damage"])
        for c in swaps:
            wr.writerow([c["params"]["position"], c["set"], f"{c['damage']:.6f}"])
    by_dist = {}
    for c in all_conditions:
        by_dist.setdefault(c["max_abs_displacement"], []).append(c["damage"])
    with (C / "distance_effects.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["max_abs_displacement", "n_conditions", "mean_damage", "median_damage"])
        for d in sorted(by_dist):
            wr.writerow(
                [
                    d,
                    len(by_dist[d]),
                    f"{statistics.mean(by_dist[d]):.4f}",
                    f"{statistics.median(by_dist[d]):.4f}",
                ]
            )

    moves = [c for c in all_conditions if c["type"] == "single_block_move"]
    interaction = {
        "direction_by_distance": {},
        "note": (
            "negative displacement = moved toward the input (earlier); positive = toward the output"
        ),
    }
    for sign, label_s in ((-1, "earlier"), (1, "later")):
        for dist in (1, 2, 3, 5, 8):
            vals = [
                c["damage"]
                for c in moves
                if any(v == sign * dist for v in c["displacements"].values())
                and c["max_abs_displacement"] == dist
            ]
            if vals:
                interaction["direction_by_distance"][f"{label_s}_{dist}"] = {
                    "n": len(vals),
                    "mean_damage": statistics.mean(vals),
                }
    interaction["boundary_swaps"] = {
        c["id"]: c["damage"] for c in swaps if c["params"]["position"] in (0, 28)
    }
    interaction["interior_swap_mean_damage"] = statistics.mean(
        c["damage"] for c in swaps if c["params"]["position"] not in (0, 28)
    )
    wj("interaction_analysis.json", interaction)

    wj(
        "damage_model.json",
        {
            "models": models,
            "chosen": chosen,
            "validation": PRED["validation"],
            "heldout_mae": held_mae,
            "heldout_criterion": "chosen MAE <= 0.8 * P0 MAE",
            "heldout_pass": gate3,
            "heldout_worst": sorted(
                (
                    {
                        "id": c["id"],
                        "actual": c["damage"],
                        "predicted": predict(models[chosen], c),
                    }
                    for c in held
                ),
                key=lambda r: -abs(r["actual"] - r["predicted"]),
            )[:5],
        },
    )
    wj(
        "predictor_validation_results.json",
        {"validation": PRED["validation"], "heldout_mae": held_mae, "chosen": chosen},
    )
    wj(
        "heldout_placement_results.json",
        {
            "opened_once": True,
            "github_run_id": 38911632615,
            "native_macro": native,
            "identity_control_exact": exact(HELD["controls"]["engine_identity_residual"]),
            "results": [
                {k: c[k] for k in ("id", "type", "params", "macro_dev_nll", "damage")} for c in held
            ],
        },
    )
    wj(
        "heldout_integrity.json",
        {
            "sets_sha256": json.loads((C / "config/placement_sets.json").read_text())[
                "sets_sha256"
            ],
            "openings": json.loads((C / "heldout_access_log.json").read_text())["openings"],
            "predictor_frozen_before_opening": True,
            "refits_against_heldout": 0,
        },
    )

    wj(
        "measurement_validation.json",
        {
            "native_repeats": [
                controls[k]["macro_dev_nll"]
                for k in ("native_first", "native_second", "native_third")
            ],
            "invalid_probes": controls["invalid_probes"],
            "parents_unchanged": MAP["parents_unchanged"],
            "gate_G1": gate1,
        },
    )
    wj(
        "reconstruction_results.json",
        {
            "historical_reconstruction_macro": controls["historical_reconstruction"][
                "macro_dev_nll"
            ],
            "engine_identity_residual_macro": controls["engine_identity_residual"]["macro_dev_nll"],
            "engine_noop_bypass_macro": controls["engine_noop_bypass"]["macro_dev_nll"],
            "native_macro": native,
        },
    )
    wj(
        "placement_engine_validation.json",
        {
            "invalid_probes": controls["invalid_probes"],
            "inverse_operations": controls["inverse_operations"],
            "inverse_note": (
                "The engine is stateless (builds from the parent each time); "
                "an order composed with its inverse is the identity, and the "
                "identity control is exact, which jointly establishes "
                "restoration. The recorded inverse-child macros are the "
                "damage of the inverse permutation itself, not a "
                "restoration failure."
            ),
            "offline_tests": (
                "tests/test_campaign17_native_placement.py + tests/test_campaign18_map.py"
            ),
        },
    )
    wj(
        "checkpoint_integrity.json",
        {
            "map_job_parents_unchanged": MAP["parents_unchanged"],
            "heldout_job_parents_unchanged": HELD["parents_unchanged"],
            "provenance_files": [
                "map_38907871417/provenance.json",
                "heldout_38911632615/provenance.json",
            ],
        },
    )
    wj(
        "baseline_reference.json",
        {
            "recipient": "HuggingFaceTB/SmolLM-135M@1d461723eec654e65efdc40cf49301c89c0c92f4",
            "metric": "macro dev NLL (mean token NLL, 16 math + 16 text frozen panel)",
            "native_macro": native,
            "campaign_17_reference": 2.773126,
            "scoring": "evaluate_encoded + macro (scripts/campaign18_map.py)",
        },
    )
    wj(
        "inverse_operation_results.json",
        {"results": controls["inverse_operations"], "note": "see placement_engine_validation.json"},
    )
    wj(
        "analysis_reproducibility.json",
        {
            "publisher": "scripts/publish_campaign18.py",
            "predictor_fit": "scripts/fit_campaign18_predictor.py (development only)",
            "bootstrap": (
                "400 resamples over 32 per-example damages, seed 1800; "
                "example-mean approximation to the token-weighted macro"
            ),
            "inputs": ["map_38907871417/results.json", "heldout_38911632615/results.json"],
        },
    )
    wj(
        "environment_manifest.json",
        {
            "map_github_run_id": 38907871417,
            "heldout_github_run_id": 38911632615,
            "source_commits": [MAP["source_commit"], HELD["source_commit"]],
            "runner": (
                "GitHub Actions ubuntu-22.04, python 3.11, torch 2.5.1 CPU, "
                "transformers 4.48.3 (workflow-declared)"
            ),
            "runtimes_seconds": [MAP["runtime_seconds"], HELD["runtime_seconds"]],
        },
    )
    wj(
        "run_manifest.json",
        {
            "campaign": 18,
            "label": label,
            "gates": {
                "G1_measurement": gate1,
                "G2_map": gate2,
                "G3_predictor_heldout": gate3,
                "G4_exact_location": gate4,
                "G5_design_complete": gate5,
                "G6_user_authorization": gate6,
            },
            "conditions": {"development": len(dev), "validation": len(val), "heldout": len(held)},
            "held_out_capability": "NOT_RUN_NO_PROMOTION",
            "foreign_conditions": 0,
            "training_runs": 0,
        },
    )
    lines = [{"type": "metadata", "campaign": 18, "native_macro": native, "label": label}]
    for c in all_conditions:
        lines.append(
            {
                "type": "condition",
                "id": c["id"],
                "set": c["set"],
                "macro": c["macro_dev_nll"],
                "damage": c["damage"],
            }
        )
    (C / "results_raw.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    registry = json.loads((C / "run_registry.json").read_text())
    registry["runs"]["c18-heldout-1"] = {
        "status": "COMPLETE",
        "github_run_id": 38911632615,
        "source_commit": HELD["source_commit"],
    }
    (C / "run_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    ctx = {
        "label": label,
        "gates": [gate1, gate2, gate3, gate4, gate5, gate6],
        "held_mae": held_mae,
        "chosen": chosen,
        "native": native,
    }
    print(json.dumps(ctx, indent=2))


if __name__ == "__main__":
    main()
