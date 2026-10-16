"""Recompute registered language-only diagnostic records. No dual-capability success inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .audit import sha256_file
from .benchdata import load_benchmarks, text_chunks
from .benchmarks import summarize_language
from .evidence import checked_path, paired_interval, read_rows


def validate_language_diagnostic(root: str | Path, *, dataset_root: str | Path | None = None):
    root = Path(root)
    result = json.loads((root / "results.json").read_text())
    config = json.loads((root / "registration.json").read_text())
    if sha256_file(root / "registration.json") != result["registration_sha256"]:
        raise ValueError("Diagnostic registration checksum mismatch")
    if result["status"] != "PASS":
        if not result.get("error"):
            raise ValueError("Incomplete language diagnostic needs a failure/interruption reason")
        return {
            "status": "PASS",
            "scope": "failed/blocked record, not completed scientific execution",
        }
    if set(result["comparisons"]) != set(config["variants"]):
        raise ValueError("Missing registered language comparison")
    expected = None
    source_text = None
    if dataset_root is not None:
        _, data = load_benchmarks(dataset_root)
        b = config["benchmark"]
        source_text = text_chunks(
            data["wikitext2"]["test"],
            characters=b["window_characters"],
            limit=b["window_offset"] + b["window_count"],
        )[b["window_offset"] :]
    observations = {}
    for name, entry in result["comparisons"].items():
        if entry["status"] != "PASS" or entry["kind"] != "measured":
            raise ValueError("Incomplete comparison cannot complete the language diagnostic")
        file = checked_path(root, entry["records"])
        if sha256_file(file) != entry["records_sha256"]:
            raise ValueError("Language observation checksum mismatch")
        rows = read_rows(file)
        if len(rows) != config["benchmark"]["window_count"]:
            raise ValueError("Language diagnostic silently omitted observations")
        ids = [(r["id"], r["text_sha256"], r["text_bytes"]) for r in rows]
        if len({r["id"] for r in rows}) != len(rows) or (expected is not None and ids != expected):
            raise ValueError("Language variants have different ordered samples")
        expected = ids
        if summarize_language(rows) != entry["metrics"]:
            raise ValueError("Substituted language metric summary")
        if source_text is not None:
            for r in rows:
                text = source_text[r["id"]]
                if (
                    hashlib.sha256(text.encode()).hexdigest() != r["text_sha256"]
                    or len(text.encode()) != r["text_bytes"]
                ):
                    raise ValueError("Language source identity/byte denominator mismatch")
        observations[name] = rows
    full = observations["published_calibrated_child"]
    for name in ("matched_random_sector", "bypass_graft", "bypass_prefix", "bypass_suffix"):
        expected_pair = paired_interval(observations[name], full, "wikitext2", seed=config["seed"])
        if result["paired_language"][name] != expected_pair:
            raise ValueError("Substituted paired diagnostic summary")
    bpb = result["comparisons"]["published_calibrated_child"]["metrics"]["bits_per_byte"]
    host = result["comparisons"]["recipient_parent"]["metrics"]["bits_per_byte"]
    enough = len(full) >= config["criteria"]["minimum_windows"]
    retained = enough and bpb <= min(
        config["criteria"]["maximum_language_bpb"],
        config["criteria"]["language_retention_ratio"] * host,
    )
    if result["language_retention"]["status"] != (
        "PASS" if retained else "FAIL"
    ) or not math.isclose(
        result["language_retention"]["child_to_recipient_bpb_ratio"], bpb / host, rel_tol=1e-12
    ):
        raise ValueError("Invented language retention verdict")
    better = enough and result["paired_language"]["matched_random_sector"]["bootstrap_95"][0] > 0
    if result["learned_sector_advantage"]["status"] != ("PASS" if better else "FAIL"):
        raise ValueError("Invented learned-sector advantage verdict")
    if (
        result["inheritance"]["status"] != "NOT_RUN"
        or result["math_benchmark"]["status"] != "NOT_RUN"
    ):
        raise ValueError("Language-only diagnosis cannot claim mathematical/dual inheritance")
    if not all(
        value is True for key, value in result["frozen_invariance"].items() if key != "status"
    ):
        raise ValueError("A frozen original/control tensor changed")
    if (
        result["fresh_reference"]["status"] != "PASS"
        or not result["fresh_reference"]["identical_logits"]
        or not result["fresh_reference"]["generation_equal"]
    ):
        raise ValueError("Reference checkpoint reconstruction was not repeatable")
    return {
        "status": "PASS",
        "scope": "registered language diagnostic consistency, not provenance authentication",
    }
