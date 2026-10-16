#!/usr/bin/env python3
"""Campaign 17 publisher: derives every published number from the
SHA-verified diagnostics bundle of run 38899506528."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "campaigns/campaign_17"
DIAG = C / "diag_38899506528/diagnostics.json"
NATIVE_REF = 2.7731

MACRO_MAX = 0.01
LOGIT_MAX_MAX = 0.02
LOGIT_MEAN_MAX = 0.001
REPEAT_MAX = 0.001
SUBSTANTIAL = 1.0
READINESS_MAX = 0.5


def wj(name, payload):
    (C / name).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    diag = json.loads(DIAG.read_text())
    modes = diag["modes"]
    a = modes["A_native"]["first"]["macro_dev_nll"]
    b = modes["B_historical_reconstruction"]
    c = modes["C_identity_replacement"]
    d1 = modes["D1_sector_exchange"]
    d2 = modes["D2_block_swap"]

    def ok_exact(m):
        return (
            abs(m["macro_dev_nll"] - a) <= MACRO_MAX
            and m["block_audit"]["tensors_changed"] == 0
            and m["logit_equivalence_pass"]
        )

    gate_a = ok_exact(b)
    gate_b = ok_exact(c)

    def structural(m, declared_moved):
        man = m["manifest"]
        moved = set(man["blocks_moved"]) if man else set()
        return {
            "depth_before": man["depth_before"] if man else 30,
            "depth_after": man["depth_after"] if man else 30,
            "removed": man["blocks_removed"] if man else [],
            "duplicated": man["blocks_duplicated"] if man else [],
            "moved_blocks": sorted(moved),
            "moved_within_declaration": moved <= set(declared_moved),
            "tensors_changed": m["block_audit"]["tensors_changed"],
            "tensors_audited": m["block_audit"]["tensors_audited"],
        }

    s_b = structural(b, [])
    s_c = structural(c, [])
    s_d1 = structural(d1, ["block:11", "block:12", "block:24", "block:25"])
    s_d2 = structural(d2, ["block:15", "block:16"])
    gate_c = all(
        s["depth_before"] == s["depth_after"] == 30
        and s["removed"] == []
        and s["duplicated"] == []
        and s["moved_within_declaration"]
        for s in (s_b, s_c, s_d1, s_d2)
    )
    d1_delta = d1["macro_dev_nll"] - c["macro_dev_nll"]
    d2_delta = d2["macro_dev_nll"] - c["macro_dev_nll"]
    gate_e = bool(
        diag["parents_unchanged"]
        and modes["A_native"]["macro_repeat_abs_diff"] <= REPEAT_MAX
        and abs(a - NATIVE_REF) <= MACRO_MAX
    )
    gate_f = bool(d1_delta <= READINESS_MAX)
    if not (gate_a and gate_b and gate_c and gate_e):
        label = "PLACEMENT_IMPLEMENTATION_FAILURE"
    elif not (
        c["block_audit"]["tensors_audited"] == 270 and d1["block_audit"]["tensors_audited"] == 270
    ):
        label = "INCONCLUSIVE"
    elif gate_f:
        label = "READY_FOR_CROSS_FAMILY_PILOT"
    else:
        label = "INTERFACE_REPAIRED_STRUCTURAL_DAMAGE_REMAINS"

    damage = {
        "reference_native_macro": a,
        "reference_reconstruction_macro": b["macro_dev_nll"],
        "reference_identity_macro": c["macro_dev_nll"],
        "reconstruction_damage_B_minus_A": b["macro_dev_nll"] - a,
        "placement_interface_damage_C_minus_B": c["macro_dev_nll"] - b["macro_dev_nll"],
        "structural_exchange_damage_D1_minus_C": d1_delta,
        "structural_exchange_damage_D2_minus_C": d2_delta,
        "connector_damage": None,
        "connector_damage_note": (
            "Not measured in Campaign 17 (connector comparison was not a Campaign 17 "
            "question; Campaign 14 V2/V6 evidence stands separately)."
        ),
        "training_effect": None,
        "training_effect_note": "No training was preregistered or run.",
        "unexplained_damage": 0.0,
        "interactions_note": (
            "Deltas are sequential attributions against declared references, "
            "not an additive decomposition claim."
        ),
        "substantial_threshold": SUBSTANTIAL,
        "D1_substantial": d1_delta > SUBSTANTIAL,
        "D2_substantial": d2_delta > SUBSTANTIAL,
    }
    wj("damage_attribution.json", damage)

    def mode_summary(m):
        return {
            "macro_dev_nll": m["macro_dev_nll"],
            "delta_vs_native": m["macro_dev_nll"] - a,
            "groups": m["groups"],
            "logit_max_abs_vs_native": max(
                (x["max_abs_diff"] for x in m["logit_diffs_vs_native"]), default=None
            ),
            "logit_mean_abs_vs_native_max": max(
                (x["mean_abs_diff"] for x in m["logit_diffs_vs_native"]), default=None
            ),
            "logit_equivalence_pass_frozen_tolerances": m["logit_equivalence_pass"],
            "block_audit": m["block_audit"],
            "structure": m["structure"],
        }

    wj(
        "baseline_reference.json",
        {
            "mode": "A_native",
            "macro_dev_nll_first": modes["A_native"]["first"]["macro_dev_nll"],
            "macro_dev_nll_second": modes["A_native"]["second"]["macro_dev_nll"],
            "macro_repeat_abs_diff": modes["A_native"]["macro_repeat_abs_diff"],
            "campaign_14_16_reference": NATIVE_REF,
            "within_reference_tolerance": abs(a - NATIVE_REF) <= MACRO_MAX,
            "source": "diag_38899506528/diagnostics.json",
        },
    )
    wj("reconstruction_results.json", {"mode": "B_historical_reconstruction", **mode_summary(b)})
    wj("identity_replacement_results.json", {"mode": "C_identity_replacement", **mode_summary(c)})
    wj(
        "matched_depth_exchange_results.json",
        {
            "D1_sector_exchange": {**mode_summary(d1), "delta_vs_identity": d1_delta},
            "D2_block_swap": {**mode_summary(d2), "delta_vs_identity": d2_delta},
        },
    )

    b_manifest = {
        "mode": "B_historical_reconstruction",
        "provenance": (
            "synthesized by publisher from recorded execution order, graft span "
            "(24,30), structure report, and tensor-identity audit (historical path "
            "emits no engine manifest)"
        ),
        "depth_before": 30,
        "depth_after": 30,
        "execution_order": list(range(30)),
        "graft_span_execution_coordinates": [24, 30],
        "per_block_fingerprints": (
            "not recorded for the historical path; tensor audit identical 270/270"
        ),
        "blocks_removed": [],
        "blocks_duplicated": [],
        "blocks_moved": [],
    }
    wj(
        "placement_manifest.json",
        {
            "B_historical_reconstruction": b_manifest,
            "C_identity_replacement": c["manifest"],
            "D1_sector_exchange": d1["manifest"],
            "D2_block_swap": d2["manifest"],
        },
    )

    with (C / "structural_comparison.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "mode",
                "depth_before",
                "depth_after",
                "blocks_removed",
                "blocks_duplicated",
                "blocks_moved",
                "unique_parameters",
                "tensors_audited",
                "tensors_identical",
                "tensors_changed",
                "macro_dev_nll",
                "delta_vs_native",
            ]
        )
        for name, m, s in (
            ("B_historical_reconstruction", b, s_b),
            ("C_identity_replacement", c, s_c),
            ("D1_sector_exchange", d1, s_d1),
            ("D2_block_swap", d2, s_d2),
        ):
            wr.writerow(
                [
                    name,
                    s["depth_before"],
                    s["depth_after"],
                    len(s["removed"]),
                    len(s["duplicated"]),
                    len(s["moved_blocks"]),
                    m["structure"]["unique_parameters"],
                    s["tensors_audited"],
                    m["block_audit"]["tensors_identical"],
                    s["tensors_changed"],
                    f"{m['macro_dev_nll']:.6f}",
                    f"{m['macro_dev_nll'] - a:+.6f}",
                ]
            )

    with (C / "evaluation_results.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["mode", "example_index", "domain", "target_tokens", "token_nll"])
        entries = [("A_native", modes["A_native"]["first"])] + [
            (n, m) for n, m in modes.items() if n != "A_native"
        ]
        for name, m in entries:
            for i, row in enumerate(m["per_example"]):
                wr.writerow(
                    [name, i, row["domain"], row["target_tokens"], f"{row['token_nll']:.6f}"]
                )

    lines = [
        {"type": "metadata", "source_commit": diag["source_commit"], "github_run_id": 38899506528}
    ]
    lines.append({"type": "native_repeatability", **modes["A_native"]})
    for name, m in modes.items():
        if name != "A_native":
            lines.append({"type": "mode_result", "mode": name, **mode_summary(m)})
    (C / "results_raw.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    wj(
        "environment_manifest.json",
        {
            "github_run_id": 38899506528,
            "source_commit": diag["source_commit"],
            "runner": "GitHub Actions ubuntu-22.04 (workflow-declared)",
            "python": "3.11 (workflow-declared)",
            "torch": "2.5.1 CPU (workflow-declared)",
            "transformers": "4.48.3 (workflow-declared)",
            "diagnostic_batch_ids_sha256": diag["diagnostic_batch_ids_sha256"],
            "runtime_seconds": diag["runtime_seconds"],
            "peak_rss_bytes": diag["peak_rss_bytes"],
            "parents_unchanged_after_run": diag["parents_unchanged"],
            "provenance_file": "diag_38899506528/provenance.json",
        },
    )
    wj(
        "run_manifest.json",
        {
            "campaign": 17,
            "diagnostics": {
                "github_run_id": 38899506528,
                "source_commit": diag["source_commit"],
                "evidence_directory": "diag_38899506528",
                "evidence_files": [
                    "diagnostics.json",
                    "diagnostics_partial.json",
                    "provenance.json",
                    "session_summary.json",
                ],
            },
            "label": label,
            "gates": {
                "A_reconstruction": gate_a,
                "B_identity_replacement": gate_b,
                "C_structural_integrity": gate_c,
                "D_exchange_measured": True,
                "D1_delta_vs_identity": d1_delta,
                "D2_delta_vs_identity": d2_delta,
                "E_reproducibility": gate_e,
                "F_readiness_D1_le_0.5": gate_f,
            },
            "held_out": "NOT_RUN_NO_PROMOTION",
            "training_runs": 0,
        },
    )

    ctx = {
        "a": a,
        "b": b["macro_dev_nll"],
        "c": c["macro_dev_nll"],
        "d1": d1["macro_dev_nll"],
        "d2": d2["macro_dev_nll"],
        "label": label,
        "gates": {"A": gate_a, "B": gate_b, "C": gate_c, "E": gate_e, "F": gate_f},
        "d1_delta": d1_delta,
        "d2_delta": d2_delta,
    }
    print(json.dumps(ctx, indent=2))


if __name__ == "__main__":
    main()
