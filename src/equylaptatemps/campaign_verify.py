"""Campaign configuration/artifact/raw summary verification, not forged-provenance detection."""

from __future__ import annotations

import json
from pathlib import Path

from .audit import sha256_file
from .campaign_training import stable_hash
from .completion import summarize_completed_math
from .evidence import checked_path, read_rows, validate_results


def validate_replay(root, dataset_root=None):
    root = Path(root)
    validate_results(root, dataset_root=dataset_root)
    result = json.loads((root / "math_replay/results.json").read_text())
    config = json.loads((root / "math_replay/registration.json").read_text())
    if result["status"] != "PASS":
        raise ValueError("Math baseline replay incomplete")
    if set(result["comparisons"]) != set(config["variants"]):
        raise ValueError("Missing replay variant")
    if sha256_file(root / "math_replay/registration.json") != result["registration_sha256"]:
        raise ValueError("Replay config changed")
    for entry in result["comparisons"].values():
        file = checked_path(root / "math_replay", entry["records"])
        if sha256_file(file) != entry["records_sha256"]:
            raise ValueError("Replay raw answer checksum changed")
        rows = read_rows(file)
        if (
            len(rows) != config["benchmark"]["sample_count"]
            or summarize_completed_math(rows) != entry["metrics"]
        ):
            raise ValueError("Replay numeric/completion summary inconsistent")
    if not result["fresh_process"]["parents_loaded"]:
        raise ValueError("This replay reconstruction required parents; do not label standalone")
    return {
        "status": "PASS",
        "scope": "baseline config/raw consistency, not malicious provenance authentication",
    }


def validate_matrix(root, *, external_artifact=None):
    """Check config and development records; absent binaries are explicit, not silently skipped."""
    root = Path(root)
    config = json.loads((root / "registration.json").read_text())
    database = json.loads((root / "campaign_02_results.json").read_text())
    if stable_hash(config["experiments"]) != database["registry_sha256"]:
        raise ValueError("Registry configuration changed")
    lookup = {e["id"]: e for e in config["experiments"]}
    if set(lookup) != set(database["records"]):
        raise ValueError("Missing campaign experiment record")
    external = None
    if external_artifact is not None:
        external = json.loads(Path(external_artifact).read_text())
        if not external.get("artifact_name") or not external.get("artifact_digest"):
            raise ValueError("External artifact receipt lacks storage identity")
    missing_binaries = []
    for experiment, record in database["records"].items():
        if stable_hash(lookup[experiment]) != record["config_sha256"]:
            raise ValueError("Experiment configuration was edited")
        run = root / "runs" / experiment
        if record["execution_status"] == "COMPLETED":
            if not record["artifacts"]:
                raise ValueError("Unverified completed experiment")
            for name, digest in record["artifacts"].items():
                path = checked_path(root, name)
                if path.exists() and sha256_file(path) != digest:
                    raise ValueError("Missing/tampered completed artifact")
                if not path.exists():
                    if external is None:
                        raise ValueError("Completed artifact was not retained locally")
                    missing_binaries.append(name)
            result_path = run / "results.json"
            if not result_path.exists():
                raise ValueError("Completed experiment has no results record")
            report = json.loads(result_path.read_text())
            if report["validation"] != record["metrics"]:
                raise ValueError("Registry/report metric mismatch")
            for group in report["validation"]["groups"].values():
                if group["token_nll"] != group["nll_sum"] / group["target_tokens"]:
                    raise ValueError("Invented validation loss")
            if report["evaluation_status"] != "NOT_RUN" or report["test_metrics"] is not None:
                raise ValueError("Development screening cannot claim final TEST results")
            if lookup[experiment]["stage"] == "calibration":
                fit = report["training"]
                if (
                    fit["execution_status"] != "COMPLETED"
                    or fit["completed_steps"] != config["training"]["steps"]
                ):
                    raise ValueError("Calibration candidate did not complete its registered budget")
                if not fit["parent_weights_unchanged"]:
                    raise ValueError("Calibration changed a frozen tensor")
        elif record["execution_status"] in {"NOT_RUN", "FAILED", "BLOCKED", "INTERRUPTED"}:
            if not record.get("reason"):
                raise ValueError("Incomplete campaign record lacks an explicit reason")
        else:
            raise ValueError("Unknown campaign execution status")
    scope = (
        "configuration, development metrics and retained artifacts; no TEST promotion is implied"
    )
    if missing_binaries:
        scope += "; candidate binaries retained only in the identified Actions artifact"
    return {"status": "PASS", "scope": scope, "locally_unverified_binaries": missing_binaries}
