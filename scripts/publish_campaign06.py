#!/usr/bin/env python3
"""Build Campaign 6 results from imported, hash-verified run records."""

import json
from pathlib import Path

ROOT = Path("campaigns/campaign_06")
RUN = ROOT / "capacity_38428594451"


def load(path):
    return json.loads(Path(path).read_text())


def macro(metrics):
    groups = metrics["groups"]
    return (groups["domain:language"]["token_nll"] + groups["domain:math"]["token_nll"]) / 2


def record(exp_id):
    data = load(RUN / "runs" / exp_id / "results.json")
    return {
        "macro_dev_nll": macro(data["validation"]),
        "language_dev_nll": data["validation"]["groups"]["domain:language"]["token_nll"],
        "math_dev_nll": data["validation"]["groups"]["domain:math"]["token_nll"],
        "trainable_parameters": data["structure"]["trainable_dense_seam_parameters"],
    }


def main():
    results = {
        "schema_version": 1,
        "project": "HETEROGENEOUS AI MODEL SURGERY",
        "campaign": 6,
        "date": "2026-10-13",
        "run": "38428594451",
        "references_random_sector": {
            "unconstrained_fullwidth_32_steps": 4.9024,
            "unconstrained_fullwidth_96_steps": 4.0137,
            "mandatory_bn128_delta_32_steps": 8.0091,
        },
        "new_runs": {
            "c06_bn256_delta_rand_42": record("c06_bn256_delta_rand_42"),
            "c06_bn128_delta_rand_96": record("c06_bn128_delta_rand_96"),
        },
        "decision_rule": "remediable if macro <= 1.10x matched-budget unconstrained reference",
        "decision": {
            "bn256_32_threshold": 5.393,
            "bn128_96_threshold": 4.415,
            "capacity_remediable": False,
            "conclusion": (
                "Widening the bottleneck (8.0091 -> 7.3232) and tripling "
                "budget (8.0091 -> 7.0706) each help, but neither approaches "
                "the unconstrained child (4.9024/4.0137). The mandatory-use "
                "pathway is intrinsically lossy at these budgets."
            ),
        },
        "outcome": {
            "classification": "C",
            "label": "Final negative for this pairing/interface family at this scale",
            "rationale": (
                "Campaigns 3-6 jointly show: unconstrained bridges ignore "
                "the donor; mandatory bridges use it but destroy too much "
                "host information; width/placement do not fix this; and "
                "capacity controls prove the forced pathway itself cannot "
                "match unconstrained performance with random sectors."
            ),
        },
        "heldout_confirmation": {"status": "NOT_RUN_NO_PROMOTION"},
    }
    (ROOT / "campaign_06_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote campaign_06_results.json")


if __name__ == "__main__":
    main()
