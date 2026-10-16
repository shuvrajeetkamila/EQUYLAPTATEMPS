"""Completion-aware math record validation; diagnostics do not redefine inheritance thresholds."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .audit import sha256_file
from .benchdata import load_benchmarks
from .completion import STOP_REASONS, complete_answer, summarize_completed_math
from .evidence import checked_path, read_rows


def registered_prompt(question, fewshots):
    instruction = (
        "Solve the arithmetic word problem. Show reasoning and finish with #### "
        "followed by the final number on its own line.\n\n"
    )
    examples = "\n\n".join(
        "Question: " + r["question"] + "\nAnswer: " + r["answer"] for r in fewshots
    )
    return instruction + examples + "\n\nQuestion: " + question + "\nAnswer: "


def validate_completed_math(root: str | Path, *, dataset_root: str | Path | None = None):
    root = Path(root)
    result = json.loads((root / "results.json").read_text())
    config = json.loads((root / "registration.json").read_text())
    if sha256_file(root / "registration.json") != result["registration_sha256"]:
        raise ValueError("Completed-math registration checksum mismatch")
    if result["status"] != "PASS":
        if not result.get("error"):
            raise ValueError("Incomplete math run must record a blocker/failure")
        return {"status": "PASS", "scope": "failed/blocked trace, not completed benchmark"}
    if set(result["comparisons"]) != set(config["variants"]):
        raise ValueError("Missing registered math comparison")
    if set(result["cache_oracles"]) != set(config["variants"]) or any(
        o["status"] != "PASS" or not o["argmax_equal"] for o in result["cache_oracles"].values()
    ):
        raise ValueError("Math generation cannot bypass numerical/token cache-oracle gates")
    if result["reverse_cache_oracle"]["status"] != "PASS":
        raise ValueError("Reverse cache oracle failed")
    selection = json.loads((root / "data-selection.json").read_text())
    data = None
    fewshots = None
    if dataset_root is not None:
        _, data = load_benchmarks(dataset_root)
        order = [
            i
            for i in range(len(data["gsm8k"]["test"]))
            if i not in config["benchmark"]["exclude_previously_observed_test_ids"]
        ]
        random.Random(config["test_seed"]).shuffle(order)
        if selection["test_ids"] != order[: config["benchmark"]["sample_count"]]:
            raise ValueError("Held-out samples differ from registered seeded selection")
        fewshots = [data["gsm8k"]["train"][i] for i in selection["fewshot_train_ids"]]
    ordered = None
    for entry in result["comparisons"].values():
        file = checked_path(root, entry["records"])
        if (
            entry["status"] != "PASS"
            or entry["kind"] != "measured"
            or sha256_file(file) != entry["records_sha256"]
        ):
            raise ValueError("Missing/tampered math observation record")
        rows = read_rows(file)
        if len(rows) != config["benchmark"]["sample_count"]:
            raise ValueError("Truncated/missing math observations cannot be silently omitted")
        identities = [(r["id"], r["question_sha256"], r["prompt_sha256"]) for r in rows]
        if len({r["id"] for r in rows}) != len(rows) or (
            ordered is not None and identities != ordered
        ):
            raise ValueError("Math variants require the same ordered questions/prompts")
        ordered = identities
        if [r["id"] for r in rows] != selection["test_ids"]:
            raise ValueError("Math record identity mismatch")
        if summarize_completed_math(rows) != entry["metrics"]:
            raise ValueError("Invented math metric summary")
        for row in rows:
            if row["stop_reason"] not in STOP_REASONS or not all(
                type(i) is int and i >= 0 for i in row["generated_ids"]
            ):
                raise ValueError("Invalid generation termination/token schema")
            if (
                len(row["generated_ids"]) > config["benchmark"]["max_new_tokens"]
                or row["input_tokens"] > config["benchmark"]["max_input_tokens"]
            ):
                raise ValueError("Generation exceeds the registered input/output budget")
            if (
                row["stop_reason"] == "max_new_tokens"
                and len(row["generated_ids"]) != config["benchmark"]["max_new_tokens"]
            ):
                raise ValueError("Budget-terminated generation has an inconsistent token count")
            if (
                row["stop_reason"] == "completed_answer"
                and complete_answer(row["response"], row["stop_reason"]) is None
            ):
                raise ValueError("Claimed complete answer has no closed numeric final marker")
            if data is not None:
                original = data["gsm8k"]["test"][row["id"]]
                if (
                    original["answer"] != row["reference"]
                    or hashlib.sha256(original["question"].encode()).hexdigest()
                    != row["question_sha256"]
                ):
                    raise ValueError("Substituted original benchmark gold/question")
                prompt = registered_prompt(original["question"], fewshots)
                if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
                    raise ValueError("Incorrect fewshot/prompt registration")
    if (
        result["inheritance"] != config["inheritance"]
        or result["inheritance"]["status"] != "NOT_RUN"
    ):
        raise ValueError(
            "Underpowered math diagnostic cannot redefine inherited-capability success"
        )
    if not all(
        value is True for key, value in result["frozen_invariance"].items() if key != "status"
    ):
        raise ValueError("Original/frozen/published child tensors changed")
    if (
        not result["fresh_process"]["generation_identical"]
        or result["fresh_process"]["parents_loaded"]
    ):
        raise ValueError("Fresh cached child-only output was not repeatable/independent")
    return {
        "status": "PASS",
        "scope": "completed-answer math consistency, not full provenance authentication",
    }
