#!/usr/bin/env python3
"""Generate Campaign 14 deliverables from SHA-verified imported records.

Gates and damage attribution are mechanical from
campaigns/campaign_14/preregistration.md (frozen before any run):
- Gate 1: native repeat |delta macro| <= 0.001.
- Gate 2: V1 (native placement, seams bypassed) all tensors identical,
  max |delta logits| <= 5e-3, |delta macro| <= 0.01.
- Gate 3: V3 (residual-identity seams at native placement) same bounds.
Damage: seams V2-V1; placement (combined, seams neutralized) V5-V3;
historical configuration initial total V6-V0; training recovery
V6->V8 and V5->V7 when the trained records are present.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path("campaigns/campaign_14")
DIAGNOSE_JOB = 38792029752
DIAGNOSE_BUNDLE_SHA = "273470b872c8df953d4cbe18d375000fc4b2c3061cd8b739f2f7cfe08be6668a"
TRAINED_DIR = None  # filled with the trained job directory when imported
TRAINED_BUNDLE_SHA = None

GATE2_3_LOGIT = 0.005
GATE2_3_MACRO = 0.01


def load(path):
    return json.loads(Path(path).read_text())


def macro_of(variant):
    if "first" in variant:
        return variant["first"]["macro_dev_nll"]
    return variant["macro_dev_nll"]


def equivalence_entry(variant, native_macro):
    diffs = variant.get("logit_diffs_vs_native") or []
    return {
        "macro_dev_nll": macro_of(variant),
        "score_degradation": macro_of(variant) - native_macro,
        "relative_degradation_percent": 100.0 * (macro_of(variant) - native_macro) / native_macro,
        "max_abs_logit_diff": max((d["max_abs_diff"] for d in diffs), default=None),
        "mean_abs_logit_diff": (
            sum(d["mean_abs_diff"] for d in diffs) / len(diffs) if diffs else None
        ),
    }


def gate23(variant, native_macro):
    eq = equivalence_entry(variant, native_macro)
    audit = variant["parameter_audit"]
    return bool(
        audit["tensors_changed"] == 0
        and eq["max_abs_logit_diff"] is not None
        and eq["max_abs_logit_diff"] <= GATE2_3_LOGIT
        and abs(eq["score_degradation"]) <= GATE2_3_MACRO
    )


def main():
    diagnostics = load(ROOT / f"diagnose_{DIAGNOSE_JOB}" / "diagnostics.json")
    provenance = load(ROOT / f"diagnose_{DIAGNOSE_JOB}" / "provenance.json")
    variants = diagnostics["variants"]
    native = macro_of(variants["V0"])

    baseline_reference = {
        "model_id": "HuggingFaceTB/SmolLM-135M",
        "revision": "1d461723eec654e65efdc40cf49301c89c0c92f4",
        "mode": provenance["mode"],
        "tokenizer_sha256": provenance["tokenizers"]["recipient_sha256"],
        "tokenizers_identical": provenance["tokenizers"]["tokenizers_identical"],
        "snapshot_files_sha256": provenance["snapshots"]["recipient"]["snapshot_files_sha256"],
        "config_files_sha256": provenance["config_files"],
        "precision": "float32",
        "threads": 2,
        "evaluation": "evaluate_encoded token NLL; GSM8K dev 16 + WikiText-2 dev 16; data seed 42",
        "diagnose_job": DIAGNOSE_JOB,
        "diagnose_bundle_sha256": DIAGNOSE_BUNDLE_SHA,
        "source_commit": diagnostics["source_commit"],
        "parents_unchanged_through_diagnostics": diagnostics["parents_unchanged"],
    }
    (ROOT / "baseline_reference.json").write_text(json.dumps(baseline_reference, indent=2) + "\n")

    v0 = variants["V0"]
    baseline_scores = {
        "macro_dev_nll": native,
        "macro_repeat_abs_diff": v0["macro_repeat_abs_diff"],
        "groups": v0["first"]["groups"],
        "per_example": v0["first"]["per_example"],
        "repeat_per_example_identical": v0["first"]["per_example"] == v0["second"]["per_example"],
    }
    (ROOT / "baseline_scores.json").write_text(json.dumps(baseline_scores, indent=2) + "\n")
    (ROOT / "baseline_outputs_manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_batch_ids_sha256": diagnostics["diagnostic_batch_ids_sha256"],
                "diagnostic_batch_examples": 4,
                "baseline_self_logit_diff": 0.0,
                "note": "Native reference logits are compared per variant in output_equivalence.json.",
            },
            indent=2,
        )
        + "\n"
    )

    gate1 = v0["macro_repeat_abs_diff"] <= 0.001
    gate2 = gate23(variants["V1"], native)
    gate3 = gate23(variants["V3"], native)
    outcome = (
        "RECONSTRUCTION_VALIDATED"
        if (gate1 and gate2 and gate3)
        else "BLOCKED" if not (gate1 and gate2) else "DAMAGE_LOCALIZED_REPAIR_UNPROVEN"
    )

    equivalence = {name: equivalence_entry(v, native) for name, v in variants.items()}
    equivalence["gates"] = {"gate1": gate1, "gate2": gate2, "gate3": gate3}
    (ROOT / "output_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n")
    (ROOT / "parameter_integrity.json").write_text(
        json.dumps(
            {name: v["parameter_audit"] for name, v in variants.items() if "parameter_audit" in v},
            indent=2,
        )
        + "\n"
    )

    trained = {}
    if TRAINED_DIR is not None:
        for run_id, variant in (("c14_v7", "V7"), ("c14_v8", "V8")):
            record = load(ROOT / TRAINED_DIR / run_id / "result.json")
            trained[variant] = {
                "macro_dev_nll": (record["validation"]["groups"]["domain:language"]["token_nll"]
                                  + record["validation"]["groups"]["domain:math"]["token_nll"]) / 2,
                "initial_macro_dev_nll": (
                    record["initial_validation"]["groups"]["domain:language"]["token_nll"]
                    + record["initial_validation"]["groups"]["domain:math"]["token_nll"]
                ) / 2,
                "job_dir": TRAINED_DIR,
                "bundle_sha256": TRAINED_BUNDLE_SHA,
            }

    damage = {
        "native_macro": native,
        "gates": {"gate1_baseline_reproducibility": gate1, "gate2_exact_reconstruction": gate2, "gate3_function_preserving_connector": gate3},
        "variants": equivalence,
        "attribution": {
            "standard_seams_at_native_placement (V2-V1)": equivalence["V2"]["score_degradation"] - equivalence["V1"]["score_degradation"],
            "campaign13_placement_with_neutral_seams (V5-V3)": equivalence["V5"]["score_degradation"] - equivalence["V3"]["score_degradation"],
            "historical_configuration_initial_total (V6-V0)": equivalence["V6"]["score_degradation"],
            "interaction_note": "Factors interact; the combined initial damage (V6) is smaller than the seams-alone damage (V2) because placement changes seam inputs. Deltas are labeled per comparison and are not assumed additive.",
        },
        "outcome": outcome,
    }
    if trained:
        damage["training_recovery"] = {
            "V6_to_V8_standard_seams": equivalence["V6"]["macro_dev_nll"] - trained["V8"]["macro_dev_nll"],
            "V5_to_V7_residual_seams": equivalence["V5"]["macro_dev_nll"] - trained["V7"]["macro_dev_nll"],
            "trained": trained,
        }
    (ROOT / "damage_localization.json").write_text(json.dumps(damage, indent=2) + "\n")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["variant", "kind", "macro_dev_nll", "score_degradation", "relative_degradation_percent", "max_abs_logit_diff", "bypass_macro", "zero_macro", "tensors_identical", "tensors_audited", "blocks_removed", "blocks_duplicated"])
    for name, v in variants.items():
        eq = equivalence[name]
        audit = v.get("parameter_audit", {})
        writer.writerow([
            name,
            v.get("kind"),
            f"{eq['macro_dev_nll']:.4f}",
            f"{eq['score_degradation']:+.4f}",
            f"{eq['relative_degradation_percent']:+.2f}",
            "" if eq["max_abs_logit_diff"] is None else f"{eq['max_abs_logit_diff']:.3e}",
            "" if v.get("bypass_macro") is None else f"{v['bypass_macro']:.4f}",
            "" if v.get("zero_macro") is None else f"{v['zero_macro']:.4f}",
            audit.get("tensors_identical", ""),
            audit.get("tensors_audited", ""),
            ";".join(map(str, audit.get("recipient_blocks_removed", []))),
            ";".join(map(str, audit.get("recipient_blocks_duplicated", []))),
        ])
    for name, t in trained.items():
        writer.writerow([name, "trained_32steps_seed42", f"{t['macro_dev_nll']:.4f}", f"{t['macro_dev_nll'] - native:+.4f}", f"{100.0 * (t['macro_dev_nll'] - native) / native:+.2f}", "", "", "", "", "", "", ""])
    (ROOT / "reconstruction_comparison.csv").write_text(buffer.getvalue())
    (ROOT / "results.json").write_text(
        json.dumps({"outcome": outcome, "native_macro": native, "damage": damage["attribution"], "trained": trained}, indent=2) + "\n"
    )
    print("outcome:", outcome, "| gates:", gate1, gate2, gate3)
    if trained:
        print("trained:", {k: round(v["macro_dev_nll"], 4) for k, v in trained.items()})


if __name__ == "__main__":
    main()
