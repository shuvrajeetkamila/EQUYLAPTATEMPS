"""Stage validation and recomputation of metric summaries; NOT fabrication-proof provenance."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .audit import sha256_file
from .benchdata import load_benchmarks, text_chunks
from .benchmarks import summarize_language, summarize_math

STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}
REQUIRED_VARIANTS = (
    "recipient_parent",
    "donor_parent",
    "calibrated_child",
    "uncalibrated_child",
    "bypass_graft",
    "bypass_prefix",
    "bypass_suffix",
    "zero_input_seam",
    "zero_output_seam",
    "random_interface",
)
TASKS = ("gsm8k", "wikitext2")


def pending(reason="Not executed") -> dict:
    return {"status": "NOT_RUN", "kind": "unavailable", "metrics": None, "reason": reason}


def comparison_template() -> dict:
    return {name: {task: pending() for task in TASKS} for name in REQUIRED_VARIANTS}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence paths must remain inside the experiment directory")
    return path


def validate_results(root: str | Path, *, dataset_root: str | Path | None = None) -> dict:
    """Fail missing/tampered summaries, partial records, invented aggregates, and unsafe paths.

    A malicious author can forge every output and recalculate every checksum. This
    detects inconsistent evidence, not proof that execution actually occurred.
    """
    root = Path(root)
    result = json.loads((root / "results.json").read_text())
    config = checked_path(root, result["registration_file"])
    if sha256_file(config) != result["registration_sha256"]:
        raise ValueError("Registered protocol/config checksum mismatch")
    for stage in result["stages"].values():
        if stage["status"] not in STATUSES:
            raise ValueError("Invalid stage status")
        if stage["status"] in {"NOT_RUN", "BLOCKED", "FAIL"} and not stage.get("reason"):
            raise ValueError("Nonpassing stage requires an explicit reason")
    registered = json.loads(config.read_text())
    datasets = None
    wiki_texts = None
    if dataset_root is not None:
        _, datasets = load_benchmarks(dataset_root)
        wiki_texts = text_chunks(
            datasets["wikitext2"]["test"],
            characters=registered["calibration"]["wiki_characters"],
            limit=result["sample_counts"]["wikitext2"],
        )
    comparisons = result["comparisons"]
    if set(comparisons) != set(REQUIRED_VARIANTS):
        raise ValueError("Missing required comparison variant")
    ordered_samples = {}
    for tasks in comparisons.values():
        if set(tasks) != set(TASKS):
            raise ValueError("Missing benchmark task")
        for task, entry in tasks.items():
            if entry["status"] not in STATUSES:
                raise ValueError("Unknown benchmark status")
            if entry["status"] != "PASS":
                if entry.get("metrics") is not None or not entry.get("reason"):
                    raise ValueError("Unexecuted/failed benchmarks must not claim complete metrics")
                continue
            if entry.get("kind") != "measured" or not isinstance(entry.get("metrics"), dict):
                raise ValueError("Passing benchmark needs observed metrics")
            path = checked_path(root, entry["records"])
            if sha256_file(path) != entry["records_sha256"]:
                raise ValueError("Benchmark observation checksum mismatch")
            rows = read_rows(path)
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("Duplicated benchmark observations")
            if datasets is not None:
                for row in rows:
                    if task == "gsm8k":
                        original = datasets["gsm8k"]["test"][row["id"]]
                        question_digest = hashlib.sha256(original["question"].encode()).hexdigest()
                        if (
                            row["reference"] != original["answer"]
                            or row["question_sha256"] != question_digest
                        ):
                            raise ValueError("Substituted held-out reference/question identity")
                    else:
                        text = wiki_texts[row["id"]]
                        if row["text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                            raise ValueError("Substituted language benchmark text")
                        if row["text_bytes"] != len(text.encode()):
                            raise ValueError("Incorrect language byte denominator")
            identities = [
                (
                    row["id"],
                    row.get("question_sha256") if task == "gsm8k" else row.get("text_sha256"),
                )
                for row in rows
            ]
            if task in ordered_samples and identities != ordered_samples[task]:
                raise ValueError("Comparison variants use different ordered benchmark samples")
            ordered_samples[task] = identities
            summary = summarize_math(rows) if task == "gsm8k" else summarize_language(rows)
            if summary != entry["metrics"]:
                raise ValueError(
                    "Fabricated/substituted metric summary does not match observations"
                )
            if len(rows) != result["sample_counts"][task]:
                raise ValueError("Missing benchmark observations or silently dropped failures")
    if result["stages"].get("benchmarks", {}).get("status") == "PASS" and any(
        entry["status"] != "PASS" for tasks in comparisons.values() for entry in tasks.values()
    ):
        raise ValueError("Incomplete comparisons cannot make benchmark stage PASS")
    if "criteria" in registered and "inheritance" in result:
        stored = result["inheritance"]
        if "structural_transplantation" in stored:
            expected = inheritance_criteria(result, root, registered["criteria"])
            if stored != expected:
                raise ValueError(
                    "Substituted inheritance verdict does not match registered evidence"
                )
        elif stored.get("status") != "NOT_RUN" or stored.get("metrics") is not None:
            raise ValueError("Incomplete study cannot claim an inheritance verdict")
    if "benchmark" in registered:
        expected_counts = {
            "gsm8k": registered["benchmark"]["gsm_test_samples"],
            "wikitext2": registered["benchmark"]["wiki_test_samples"],
        }
        if result["sample_counts"] != expected_counts:
            raise ValueError("Reported sample counts differ from registered protocol")
    return {
        "status": "PASS",
        "scope": "internal consistency and checksums only",
        "warning": "Checksums cannot authenticate maliciously forged complete evidence",
    }


def paired_interval(left: list[dict], right: list[dict], task: str, *, seed=42, repetitions=2000):
    if not left or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Paired benchmark comparisons require identical ordered samples")
    if task == "gsm8k":
        differences = [
            int(a["correct"]) - int(b["correct"]) for a, b in zip(left, right, strict=True)
        ]
        weights = [1] * len(left)
    else:
        if any(a["text_sha256"] != b["text_sha256"] for a, b in zip(left, right, strict=True)):
            raise ValueError("Paired language samples differ")
        differences = [
            (a["nll_sum"] - b["nll_sum"]) / math.log(2) for a, b in zip(left, right, strict=True)
        ]
        weights = [row["text_bytes"] for row in left]
    generator = random.Random(seed)
    values = []
    for _ in range(repetitions):
        indices = [generator.randrange(len(left)) for _ in left]
        values.append(sum(differences[i] for i in indices) / sum(weights[i] for i in indices))
    values.sort()
    return {
        "difference": sum(differences) / sum(weights),
        "bootstrap_95": [values[int(0.025 * repetitions)], values[int(0.975 * repetitions)]],
        "resamples": repetitions,
        "seed": seed,
        "limitation": "Single run; WikiText windows may be correlated. Not an independent-seed CI.",
    }


def inheritance_criteria(result: dict, root: Path, criteria: dict) -> dict:
    """Registered capability criteria, never a forward-pass/calibration-loss success claim."""
    output = {
        name: pending("Required held-out comparisons not complete")
        for name in (
            "structural_transplantation",
            "source_capabilities",
            "functional_contribution",
            "retention",
            "complementarity",
            "improvement",
        )
    }
    output["structural_transplantation"] = {
        "status": result["stages"]["reload"]["status"],
        "kind": "derived",
        "reason": "Execution/reload/frozen-sector checks",
    }
    required = result["comparisons"]
    if any(entry["status"] != "PASS" for tasks in required.values() for entry in tasks.values()):
        return output
    if any(count < criteria["minimum_samples"] for count in result["sample_counts"].values()):
        for name in output:
            if name != "structural_transplantation":
                output[name] = pending(
                    "Exploratory pilot is below the preregistered sample minimum"
                )
        return output

    def metric(role, task, key):
        return required[role][task]["metrics"][key]

    def rows(role, task):
        return read_rows(checked_path(root, required[role][task]["records"]))

    def verdict(passed, detail):
        return {"status": "PASS" if passed else "FAIL", "kind": "derived", "reason": detail}

    donor_accuracy = metric("donor_parent", "gsm8k", "accuracy")
    host_bpb = metric("recipient_parent", "wikitext2", "bits_per_byte")
    child_accuracy = metric("calibrated_child", "gsm8k", "accuracy")
    child_bpb = metric("calibrated_child", "wikitext2", "bits_per_byte")
    source_ok = (
        donor_accuracy >= criteria["minimum_math_accuracy"]
        and host_bpb <= criteria["maximum_language_bpb"]
    )
    output["source_capabilities"] = verdict(
        source_ok, "Roles must be supported by actual original-parent scores"
    )
    retention = (
        source_ok
        and child_accuracy
        >= max(
            criteria["minimum_math_accuracy"], criteria["math_retention_fraction"] * donor_accuracy
        )
        and child_bpb <= criteria["language_retention_ratio"] * host_bpb
    )
    output["retention"] = verdict(
        retention, "Registered donor math floor + recipient language BPB ceiling"
    )
    math_interval = paired_interval(
        rows("calibrated_child", "gsm8k"), rows("bypass_graft", "gsm8k"), "gsm8k"
    )
    prefix_interval = paired_interval(
        rows("bypass_prefix", "wikitext2"), rows("calibrated_child", "wikitext2"), "wikitext2"
    )
    suffix_interval = paired_interval(
        rows("bypass_suffix", "wikitext2"), rows("calibrated_child", "wikitext2"), "wikitext2"
    )
    functional = all(
        interval["bootstrap_95"][0] > 0
        for interval in (math_interval, prefix_interval, suffix_interval)
    )
    output["functional_contribution"] = verdict(
        functional, "Positive paired math and both recipient-block language contributions"
    )
    output["functional_contribution"]["intervals"] = {
        "math": math_interval,
        "prefix": prefix_interval,
        "suffix": suffix_interval,
    }
    output["complementarity"] = verdict(
        retention and functional,
        "Both retention and role-specific functional contribution required",
    )
    improvement = child_accuracy > max(
        metric("recipient_parent", "gsm8k", "accuracy"), donor_accuracy
    ) and child_bpb < min(host_bpb, metric("donor_parent", "wikitext2", "bits_per_byte"))
    if improvement:
        math_parent = max(
            ("recipient_parent", "donor_parent"), key=lambda role: metric(role, "gsm8k", "accuracy")
        )
        language_parent = min(
            ("recipient_parent", "donor_parent"),
            key=lambda role: metric(role, "wikitext2", "bits_per_byte"),
        )
        math_improvement = paired_interval(
            rows("calibrated_child", "gsm8k"), rows(math_parent, "gsm8k"), "gsm8k"
        )
        language_improvement = paired_interval(
            rows(language_parent, "wikitext2"), rows("calibrated_child", "wikitext2"), "wikitext2"
        )
        improvement = (
            min(math_improvement["bootstrap_95"][0], language_improvement["bootstrap_95"][0]) > 0
        )
    output["improvement"] = verdict(
        improvement, "Positive paired improvement against both best-parent measures"
    )
    output["claim"] = "Conditional protocol-specific criteria, not universal superiority"
    return output
