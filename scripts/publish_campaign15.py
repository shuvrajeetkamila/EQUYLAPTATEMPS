#!/usr/bin/env python3
"""Generate Campaign 15 deliverables from SHA-verified imported records.

Gates and outcome are mechanical from campaigns/campaign_15/
preregistration.md and decision_rules.md (frozen before the panel):
delta_t(seed) = old_macro(t) - new_macro(t); positive favors new.
Precedence: IMPLEMENTATION_FAILURE > INCONCLUSIVE >
TRAINING_DEPENDENT > REPAIRED_START_ADVANTAGE > INITIALIZATION_ONLY
> NO_RELIABLE_ADVANTAGE.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path

ROOT = Path("campaigns/campaign_15")
REGRESSION_DIR = "regression_38864353328"
REGRESSION_BUNDLE_SHA = "5c4b90097732f0a3f60784c87532ce839eb82025d9a8f1ea4dd02a1c0050851e"
PANELS = {
    42: ("panel42_38865626143", "e54792ab540dcd8e8fd7628396e53c04cb480724252873df2de68d2ef6178e5f"),
    123: (
        "panel123_38865628701",
        "63f46a7a0f67b463088e59739e6b15d5c87d6ea3692df4cb783226a6018f2c62",
    ),
    31415: (
        "panel31415_38865631168",
        "a1b5622d7c4ca04384b74692d0cf09c1112b26580369567f31015b8e97fcd4fe",
    ),
}
SEEDS = (42, 123, 31415)
STEPS = (0, 8, 16, 32, 64, 96)


def load(path):
    return json.loads(Path(path).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def gate_a(diagnostics):
    v = diagnostics["variants"]
    native = v["V0"]["first"]["macro_dev_nll"]
    checks = {
        "native_repeat_abs_diff": v["V0"]["macro_repeat_abs_diff"],
        "repeat_pass": v["V0"]["macro_repeat_abs_diff"] <= 0.001,
        "parents_unchanged": diagnostics["parents_unchanged"],
    }
    for name in ("V1", "V3"):
        x = v[name]
        checks[name] = {
            "macro_abs_diff": abs(x["macro_dev_nll"] - native),
            "max_abs_logit_diff": max(e["max_abs_diff"] for e in x["logit_diffs_vs_native"]),
            "tensors_changed": x["parameter_audit"]["tensors_changed"],
        }
        checks[name]["pass"] = bool(
            checks[name]["macro_abs_diff"] <= 0.01
            and checks[name]["max_abs_logit_diff"] <= 0.005
            and checks[name]["tensors_changed"] == 0
        )
    checks["pass"] = bool(
        checks["repeat_pass"]
        and checks["parents_unchanged"]
        and checks["V1"]["pass"]
        and checks["V3"]["pass"]
    )
    return checks


def paired_stats(deltas):
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    half = 4.303 * sd / math.sqrt(n) if n > 1 else 0.0  # t(0.975, df=2)
    return {
        "deltas": deltas,
        "mean": mean,
        "median": statistics.median(deltas),
        "sd": sd,
        "favors_new": sum(1 for d in deltas if d > 0),
        "favors_old": sum(1 for d in deltas if d < 0),
        "paired_t_interval_95_df2": [mean - half, mean + half],
    }


def decide(gate_a_pass, complete, gate_b_pass, deltas32, deltas96):
    """Mechanical outcome per decision_rules.md precedence."""
    if not gate_a_pass:
        return "IMPLEMENTATION_FAILURE"
    if not complete:
        return "INCONCLUSIVE"
    mean32, mean96 = statistics.fmean(deltas32), statistics.fmean(deltas96)
    flip = (mean32 > 0) != (mean96 > 0) and mean32 != 0 and mean96 != 0
    consistent_flip = (all(d > 0 for d in deltas32) and all(d < 0 for d in deltas96)) or (
        all(d < 0 for d in deltas32) and all(d > 0 for d in deltas96)
    )
    if flip or consistent_flip:
        return "TRAINING_DEPENDENT"
    if gate_b_pass and all(d > 0 for d in deltas32):
        return "REPAIRED_START_ADVANTAGE"
    if gate_b_pass:
        return "INITIALIZATION_ONLY"
    return "NO_RELIABLE_ADVANTAGE"


def main():
    diagnostics = load(ROOT / REGRESSION_DIR / "diagnostics.json")
    gate_a_result = gate_a(diagnostics)
    (ROOT / "reconstruction_regression.json").write_text(
        json.dumps(
            {
                "job": REGRESSION_DIR,
                "bundle_sha256": REGRESSION_BUNDLE_SHA,
                "gate_a": gate_a_result,
            },
            indent=2,
        )
        + "\n"
    )
    native = diagnostics["variants"]["V0"]["first"]["macro_dev_nll"]
    (ROOT / "baseline_reference.json").write_text(
        json.dumps(
            {
                "model_id": "HuggingFaceTB/SmolLM-135M",
                "revision": "1d461723eec654e65efdc40cf49301c89c0c92f4",
                "native_macro_dev_nll": native,
                "source": f"{REGRESSION_DIR}/diagnostics.json (fresh on Campaign 15 code state)",
                "bundle_sha256": REGRESSION_BUNDLE_SHA,
            },
            indent=2,
        )
        + "\n"
    )

    scores = {}  # (condition, seed) -> {step: macro}
    raw_rows = []
    init_rows = []
    complete = True
    for seed in SEEDS:
        if seed not in PANELS:
            complete = False
            continue
        panel_dir, bundle_sha = PANELS[seed]
        for condition in ("new", "old"):
            record = load(ROOT / panel_dir / "runs" / f"c15_{condition}_{seed}" / "results.json")
            evals = {e["step"]: macro(e["validation"]) for e in record["checkpoint_evaluations"]}
            if set(evals) != set(STEPS) or record["training"]["execution_status"] != "COMPLETED":
                complete = False
            if abs(evals[96] - macro(record["validation"])) > 1e-9:
                complete = False
            scores[(condition, seed)] = evals
            for step in STEPS:
                raw_rows.append(
                    {
                        "run_id": f"c15_{condition}_{seed}",
                        "condition": condition,
                        "seed": seed,
                        "step": step,
                        "macro_dev_nll": evals[step],
                        "job_dir": panel_dir,
                        "bundle_sha256": bundle_sha,
                    }
                )
            init_rows.append(
                {
                    "run_id": f"c15_{condition}_{seed}",
                    "condition": condition,
                    "seed": seed,
                    "step0_macro": evals[0],
                    "initial_graft_mean_relative_update": record["graft_delta"][
                        "mean_relative_update"
                    ],
                    "trainable_parameters": record["structure"]["trainable_dense_seam_parameters"],
                    "parent_weights_frozen": record["parent_weights_frozen"],
                }
            )

    with (ROOT / "results_raw.jsonl").open("w") as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")
    with (ROOT / "initialization_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(init_rows[0]) if init_rows else ["run_id"])
        writer.writeheader()
        writer.writerows(init_rows)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["run_id", "condition", "seed", *[f"step_{s}" for s in STEPS]])
    for (condition, seed), evals in sorted(scores.items()):
        writer.writerow(
            [f"c15_{condition}_{seed}", condition, seed, *[f"{evals[s]:.4f}" for s in STEPS]]
        )
    (ROOT / "checkpoint_results.csv").write_text(buffer.getvalue())

    deltas = {t: [scores[("old", s)][t] - scores[("new", s)][t] for s in SEEDS] for t in STEPS}
    delta0 = deltas[0]
    gate_b = bool(all(d > 0 for d in delta0) and statistics.fmean(delta0) >= 1.0)
    outcome = decide(gate_a_result["pass"], complete, gate_b, deltas[32], deltas[96])

    by_seed = io.StringIO()
    writer = csv.writer(by_seed)
    writer.writerow(
        [
            "seed",
            "old_step0",
            "new_step0",
            "old_step32",
            "new_step32",
            "delta32",
            "old_step96",
            "new_step96",
            "delta96",
            "old_improvement_0_32",
            "new_improvement_0_32",
        ]
    )
    for s in SEEDS:
        o, n = scores[("old", s)], scores[("new", s)]
        writer.writerow(
            [
                s,
                f"{o[0]:.4f}",
                f"{n[0]:.4f}",
                f"{o[32]:.4f}",
                f"{n[32]:.4f}",
                f"{o[32] - n[32]:+.4f}",
                f"{o[96]:.4f}",
                f"{n[96]:.4f}",
                f"{o[96] - n[96]:+.4f}",
                f"{o[0] - o[32]:.4f}",
                f"{n[0] - n[32]:.4f}",
            ]
        )
    (ROOT / "results_by_seed.csv").write_text(by_seed.getvalue())

    analysis = {
        "native_macro": native,
        "gate_a_pass": gate_a_result["pass"],
        "gate_b_pass": gate_b,
        "gate_c_pass": bool(all(d > 0 for d in deltas[32])),
        "complete_panel": complete,
        "deltas_by_step": {str(t): paired_stats(deltas[t]) for t in STEPS},
        "outcome": outcome,
    }
    (ROOT / "paired_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ROOT / "results.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print("outcome:", outcome, "| Gate B:", gate_b, "| delta32:", [round(d, 4) for d in deltas[32]])


if __name__ == "__main__":
    main()
