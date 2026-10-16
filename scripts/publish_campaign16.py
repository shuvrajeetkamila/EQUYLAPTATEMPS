#!/usr/bin/env python3
"""Generate Campaign 16 deliverables from SHA-verified imported records.

Frozen rules: campaigns/campaign_16/decision_rules.md (+ Amendment 01
for the Gate A logit proxy bound). delta = old_macro - new_macro per
(seed, lr, step); positive favors the new connector.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from pathlib import Path

ROOT = Path("campaigns/campaign_16")
REGRESSION_DIR = "regression2_38874299399"
REGRESSION_BUNDLE_SHA = "cfb07ce3554b32ea86e4c84141f77c3c75ba29dec8b46230662ad8337750cf9a"
PANELS = {
    "low_0.3x": (
        "lrlow_38882548685",
        "add332cc678ad1d478f84eec07f26e5915a3561d6cb9e26bee5474f8ee466d3e",
    ),
    "base_1x": (
        "lrbase_38885462762",
        "fe26b3e472c56f3e68afd2d798cb9df4d917d64099c468cde89d73ec2f946d4c",
    ),
    "high_3x": (
        "lrhigh_38888327387",
        "05a5fdd8e2fc908c6be6d684d066a788a672b24d01a0e2587699f10969eb5ba9",
    ),
}
LR_LEVELS = ("low_0.3x", "base_1x", "high_3x")
SEEDS = (42, 123, 31415)
STEPS = (0, 8, 16, 32, 64, 96)
THRESH = 0.25


def load(path):
    return json.loads(Path(path).read_text())


def macro(metrics):
    g = metrics["groups"]
    return (g["domain:language"]["token_nll"] + g["domain:math"]["token_nll"]) / 2


def ranking(deltas):
    mean = statistics.fmean(deltas)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if mean >= THRESH and pos >= 2:
        return "NEW"
    if mean <= -THRESH and neg >= 2:
        return "OLD"
    return "TIE"


def decide(gate_a, gate_b, complete, rankings, mechanism):
    """rankings: {(lr, step): NEW/OLD/TIE}. Frozen precedence."""
    if not (gate_a and gate_b):
        return "IMPLEMENTATION_FAILURE"
    if not complete:
        return "INCONCLUSIVE"
    if all(rankings[(lr, t)] == "TIE" for lr in LR_LEVELS for t in (32, 96)):
        return "NO_RELIABLE_DIFFERENCE"
    at32 = {rankings[(lr, 32)] for lr in LR_LEVELS}
    if "NEW" in at32 and "OLD" in at32:
        return "OPTIMIZATION_MECHANISM_SUPPORTED" if mechanism else "LEARNING_RATE_SENSITIVE"
    if len(at32 - {"TIE"}) > 1 or (len(at32) > 1 and "TIE" not in at32):
        return "LEARNING_RATE_SENSITIVE"
    if len(at32) > 1:
        return "LEARNING_RATE_SENSITIVE"
    flips = sum(
        1
        for lr in LR_LEVELS
        if rankings[(lr, 32)] != rankings[(lr, 96)]
        and {rankings[(lr, 32)], rankings[(lr, 96)]} == {"NEW", "OLD"}
    )
    if flips >= 2:
        return "TRAINING_BUDGET_DEPENDENT"
    return "CONNECTOR_DIFFERENCE_PERSISTS"


def main():
    diagnostics = load(ROOT / REGRESSION_DIR / "diagnostics.json")
    v = diagnostics["variants"]
    native = v["V0"]["first"]["macro_dev_nll"]
    gate_a_checks = {"native_repeat_abs_diff": v["V0"]["macro_repeat_abs_diff"]}
    gate_a_ok = v["V0"]["macro_repeat_abs_diff"] <= 0.001 and diagnostics["parents_unchanged"]
    for name in ("V1", "V3"):
        x = v[name]
        entry = {
            "macro_abs_diff": abs(x["macro_dev_nll"] - native),
            "max_abs_logit_diff": max(e["max_abs_diff"] for e in x["logit_diffs_vs_native"]),
            "mean_abs_logit_diff": statistics.fmean(
                e["mean_abs_diff"] for e in x["logit_diffs_vs_native"]
            ),
            "tensors_changed": x["parameter_audit"]["tensors_changed"],
        }
        entry["pass"] = bool(
            entry["macro_abs_diff"] <= 0.01
            and entry["max_abs_logit_diff"] <= 0.02  # Amendment 01
            and entry["mean_abs_logit_diff"] <= 0.001  # Amendment 01
            and entry["tensors_changed"] == 0
        )
        gate_a_checks[name] = entry
        gate_a_ok = gate_a_ok and entry["pass"]
    (ROOT / "reconstruction_regression.json").write_text(
        json.dumps(
            {
                "job": REGRESSION_DIR,
                "bundle_sha256": REGRESSION_BUNDLE_SHA,
                "gate_a_pass": gate_a_ok,
                "checks": gate_a_checks,
                "attempt1": (
                    "regression_38872585083 FAILED under the original 5e-3 logit bound "
                    "(Amendment 01, failure F-C16-001); attempt 2 passes the amended bound."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / "baseline_reference.json").write_text(
        json.dumps(
            {
                "model_id": "HuggingFaceTB/SmolLM-135M",
                "revision": "1d461723eec654e65efdc40cf49301c89c0c92f4",
                "native_macro_dev_nll": native,
                "placement_only_macro_dev_nll": 12.015005577316103,
                "source": f"{REGRESSION_DIR}/diagnostics.json + Campaign 14/15 decomposition",
            },
            indent=2,
        )
        + "\n"
    )

    scores = {}  # (lr, condition, seed) -> {step: macro}
    diag_rows = []
    raw_rows = []
    traces = {}
    complete = True
    gate_b = True
    for lr in LR_LEVELS:
        if lr not in PANELS:
            complete = False
            continue
        panel_dir, bundle_sha = PANELS[lr]
        tag = {"low_0.3x": "0.3x", "base_1x": "1x", "high_3x": "3x"}[lr]
        for condition in ("old", "new"):
            for seed in SEEDS:
                run_id = f"c16_{condition}_lr{tag}_s{seed}"
                rec = load(ROOT / panel_dir / "runs" / run_id / "results.json")
                evals = {e["step"]: macro(e["validation"]) for e in rec["checkpoint_evaluations"]}
                if set(evals) != set(STEPS) or rec["training"]["execution_status"] != "COMPLETED":
                    complete = False
                if not (
                    rec["parent_weights_frozen"]
                    and rec["training"]["completed_steps"] == 96
                    and rec["structure"]["trainable_dense_seam_parameters"] == 667008
                ):
                    gate_b = False
                scores[(lr, condition, seed)] = evals
                traces[(lr, condition, seed)] = [
                    tuple(e["record_indices"]) for e in rec["training"]["trace"]
                ]
                for step in STEPS:
                    raw_rows.append(
                        {
                            "run_id": run_id,
                            "condition": condition,
                            "lr_level": lr,
                            "learning_rate": rec["learning_rate"],
                            "seed": seed,
                            "step": step,
                            "macro_dev_nll": evals[step],
                            "gap_from_native": evals[step] - native,
                            "improvement_from_step0": evals[0] - evals[step],
                            "job_dir": panel_dir,
                            "bundle_sha256": bundle_sha,
                        }
                    )
                trace = rec["training"]["trace"]
                norms = [e["gradient_norm_before_clip"] for e in trace]

                def seam_norm(window, group):
                    key = "input_seam_grad_l2" if group == "input_seam" else "output_seam_grad_l2"
                    return statistics.median(e[key] for e in window)

                last_diag = rec["checkpoint_diagnostics"][-1]
                first_diag = rec["checkpoint_diagnostics"][0]
                diag_rows.append(
                    {
                        "run_id": run_id,
                        "condition": condition,
                        "lr_level": lr,
                        "learning_rate": rec["learning_rate"],
                        "seed": seed,
                        "median_grad_norm_steps1_32": statistics.median(norms[:32]),
                        "median_grad_norm_steps33_96": statistics.median(norms[32:]),
                        "clip_fraction": sum(1 for x in norms if x > 1.0) / len(norms),
                        "median_input_seam_grad_steps1_32": seam_norm(trace[:32], "input_seam"),
                        "median_output_seam_grad_steps1_32": seam_norm(trace[:32], "output_seam"),
                        "final_update_l2_input_seam": last_diag["seam_updates"]["input_seam"][
                            "update_l2"
                        ],
                        "final_update_l2_output_seam": last_diag["seam_updates"]["output_seam"][
                            "update_l2"
                        ],
                        "fraction_parameters_changed_final": last_diag[
                            "fraction_parameters_changed"
                        ],
                        "graft_input_rms_step0": first_diag["graft_activity"]["mean_input_rms"],
                        "graft_output_rms_step0": first_diag["graft_activity"]["mean_output_rms"],
                        "graft_input_rms_step96": last_diag["graft_activity"]["mean_input_rms"],
                        "graft_output_rms_step96": last_diag["graft_activity"]["mean_output_rms"],
                    }
                )
    # Paired data order across the six cells of each seed (Gate B).
    for seed in SEEDS:
        reference = traces.get(("base_1x", "new", seed))
        for lr in LR_LEVELS:
            for condition in ("old", "new"):
                if traces.get((lr, condition, seed)) != reference:
                    gate_b = False

    with (ROOT / "results_raw.jsonl").open("w") as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")
    with (ROOT / "optimization_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0]) if diag_rows else ["run_id"])
        writer.writeheader()
        writer.writerows(diag_rows)

    deltas = {
        (lr, t): [scores[(lr, "old", s)][t] - scores[(lr, "new", s)][t] for s in SEEDS]
        for lr in LR_LEVELS
        for t in STEPS
    }
    rankings = {(lr, t): ranking(deltas[(lr, t)]) for lr in LR_LEVELS for t in STEPS}

    # Gate E: gradient ratio (new/old) at baseline LR + LR-response check.
    grad = {
        (c, s): next(
            r["median_grad_norm_steps1_32"]
            for r in diag_rows
            if r["condition"] == c and r["seed"] == s and r["lr_level"] == "base_1x"
        )
        for c in ("old", "new")
        for s in SEEDS
    }
    ratios = [grad[("new", s)] / grad[("old", s)] for s in SEEDS]
    same_side = all(r < 1 for r in ratios) or all(r > 1 for r in ratios)
    med_ratio = statistics.median(ratios)
    outside = med_ratio < 0.8 or med_ratio > 1.25
    mean32 = {lr: statistics.fmean(deltas[(lr, 32)]) for lr in LR_LEVELS}
    if same_side and outside and med_ratio < 1:
        prediction = mean32["high_3x"] >= mean32["low_0.3x"] + 0.25
    elif same_side and outside:
        prediction = mean32["low_0.3x"] >= mean32["high_3x"] + 0.25
    else:
        prediction = False
    mechanism = bool(same_side and outside and prediction)

    outcome = decide(gate_a_ok, gate_b, complete, rankings, mechanism)

    # Gate C: baseline-LR cells vs Campaign 15 records.
    c15 = {}
    for seed, d in (
        (42, "panel42_38865626143"),
        (123, "panel123_38865628701"),
        (31415, "panel31415_38865631168"),
    ):
        for condition in ("old", "new"):
            rec = load(
                Path("campaigns/campaign_15")
                / d
                / "runs"
                / f"c15_{condition}_{seed}"
                / "results.json"
            )
            c15[(condition, seed)] = {
                e["step"]: macro(e["validation"]) for e in rec["checkpoint_evaluations"]
            }
    gate_c_max = 0.0
    for seed in SEEDS:
        for condition in ("old", "new"):
            for t in STEPS:
                gate_c_max = max(
                    gate_c_max,
                    abs(scores[("base_1x", condition, seed)][t] - c15[(condition, seed)][t]),
                )

    sensitivity = {
        "rankings": {f"{lr}@{t}": rankings[(lr, t)] for lr in LR_LEVELS for t in STEPS},
        "mean_delta32_by_lr": mean32,
        "gate_e": {
            "grad_norm_ratios_new_over_old_base_lr": ratios,
            "median_ratio": med_ratio,
            "same_side_all_seeds": same_side,
            "outside_band_0.8_1.25": outside,
            "lr_response_matches_prediction": prediction,
            "mechanism_supported": mechanism,
        },
    }
    (ROOT / "learning_rate_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")

    def stats(ds):
        return {
            "deltas": ds,
            "mean": statistics.fmean(ds),
            "median": statistics.median(ds),
            "sd": statistics.stdev(ds) if len(ds) > 1 else 0.0,
            "favors_new": sum(1 for d in ds if d > 0),
            "favors_old": sum(1 for d in ds if d < 0),
        }

    analysis = {
        "native_macro": native,
        "gate_a_pass": gate_a_ok,
        "gate_b_pass": gate_b,
        "gate_c_max_abs_deviation_vs_campaign15": gate_c_max,
        "gate_c_pass": gate_c_max <= 0.01,
        "complete_matrix": complete,
        "rankings": sensitivity["rankings"],
        "mechanism_supported": mechanism,
        "deltas": {f"{lr}@{t}": stats(deltas[(lr, t)]) for lr in LR_LEVELS for t in STEPS},
        "outcome": outcome,
    }
    (ROOT / "paired_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ROOT / "results.json").write_text(json.dumps(analysis, indent=2) + "\n")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["run_id", "condition", "lr_level", "seed", *[f"step_{s}" for s in STEPS]])
    for (lr, condition, seed), evals in sorted(scores.items()):
        tag = {"low_0.3x": "0.3x", "base_1x": "1x", "high_3x": "3x"}[lr]
        writer.writerow(
            [
                f"c16_{condition}_lr{tag}_s{seed}",
                condition,
                lr,
                seed,
                *[f"{evals[s]:.4f}" for s in STEPS],
            ]
        )
    (ROOT / "checkpoint_results.csv").write_text(buffer.getvalue())

    by_seed = io.StringIO()
    writer = csv.writer(by_seed)
    writer.writerow(
        [
            "lr_level",
            "seed",
            "old_step32",
            "new_step32",
            "delta32",
            "old_step96",
            "new_step96",
            "delta96",
        ]
    )
    for lr in LR_LEVELS:
        for s in SEEDS:
            o, n = scores[(lr, "old", s)], scores[(lr, "new", s)]
            writer.writerow(
                [
                    lr,
                    s,
                    f"{o[32]:.4f}",
                    f"{n[32]:.4f}",
                    f"{o[32] - n[32]:+.4f}",
                    f"{o[96]:.4f}",
                    f"{n[96]:.4f}",
                    f"{o[96] - n[96]:+.4f}",
                ]
            )
    (ROOT / "results_by_seed.csv").write_text(by_seed.getvalue())
    print("outcome:", outcome, "| GateC max dev:", round(gate_c_max, 6), "| mechanism:", mechanism)
    print(
        "rankings@32:",
        {lr: rankings[(lr, 32)] for lr in LR_LEVELS},
        "| @96:",
        {lr: rankings[(lr, 96)] for lr in LR_LEVELS},
    )


if __name__ == "__main__":
    main()
