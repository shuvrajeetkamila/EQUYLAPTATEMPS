#!/usr/bin/env python3
"""Campaign 18 predictor freeze: fit P0/P1/P2 on development, select
on validation, write the frozen predictor, and record the held-out
opening. Held-out outcomes are not read by this script."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "campaigns/campaign_18"
MAP = C / "map_38907871417/results.json"


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


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def main():
    report = json.loads(MAP.read_text())
    native = report["controls"]["native_first"]["macro_dev_nll"]
    by_set = {"development": [], "validation": []}
    for cond in report["conditions"]:
        bucket = "development" if cond["id"].startswith("dev-") else "validation"
        by_set[bucket].append({**cond, "damage": cond["macro_dev_nll"] - native})
    dev, val = by_set["development"], by_set["validation"]

    p0 = float(np.mean([c["damage"] for c in dev]))
    models = {"P0_constant": {"kind": "constant", "value": p0}}
    for name in ("P1", "P2"):
        x = np.array([features(c)[name] for c in dev])
        y = np.array([c["damage"] for c in dev])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        models[{"P1": "P1_distance_only", "P2": "P2_position_aware"}[name]] = {
            "kind": "ols",
            "feature_set": name,
            "coefficients": [float(v) for v in coef],
        }

    def predict(model, cond):
        if model["kind"] == "constant":
            return model["value"]
        x = features(cond)[model["feature_set"]]
        return float(np.dot(model["coefficients"], x))

    validation = {}
    for name, model in models.items():
        validation[name] = {
            "dev_mae": mae([predict(model, c) for c in dev], [c["damage"] for c in dev]),
            "val_mae": mae([predict(model, c) for c in val], [c["damage"] for c in val]),
        }
    order = ["P0_constant", "P1_distance_only", "P2_position_aware"]
    chosen = min(order, key=lambda n: (validation[n]["val_mae"], order.index(n)))
    frozen = {
        "campaign": 18,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_set": "development (48)",
        "selection_set": "validation (5)",
        "models": models,
        "validation": validation,
        "chosen": chosen,
        "heldout_success_rule": "chosen heldout MAE <= 0.8 * P0_constant heldout MAE",
        "heldout_status": "SEALED_AT_FREEZE_TIME",
    }
    (C / "damage_predictor.json").write_text(json.dumps(frozen, indent=2) + "\n")

    log = json.loads((C / "heldout_access_log.json").read_text())
    log["openings"].append(
        {
            "opened_at_utc": frozen["frozen_at_utc"],
            "purpose": "single frozen-predictor confirmation (c18-heldout-1)",
            "predictor_chosen": chosen,
            "results_inspected_before_freeze": False,
        }
    )
    (C / "heldout_access_log.json").write_text(json.dumps(log, indent=2) + "\n")

    registry = json.loads((C / "run_registry.json").read_text())
    registry["runs"]["c18-map-1"] = {
        "status": "COMPLETE",
        "github_run_id": 38907871417,
        "source_commit": report["source_commit"],
    }
    registry["runs"]["c18-heldout-1"] = {"status": "OPENED_ONCE_PREDICTOR_FROZEN"}
    registry["heldout_access"] = log["openings"]
    (C / "run_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(json.dumps({"chosen": chosen, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
