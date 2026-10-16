"""Completion-aware GSM numeric scoring; never credit an arbitrary intermediate number."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .benchmarks import normalize_number, numeric_answer, wilson

VALUE = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d[\d,]*(?:\.\d+)?)?"
TAG = re.compile(r"####[ \t]*\$?[ \t]*(" + VALUE + r")[ \t]*[.!]?[ \t]*(\r?\n|$)")
BOX = re.compile(r"\\boxed\{[ \t]*\$?[ \t]*(" + VALUE + r")[ \t]*\}")
STOP_REASONS = {"eos", "max_new_tokens", "completed_answer"}


def scalar(value: str) -> str | None:
    if "/" not in value:
        return normalize_number(value)
    try:
        numerator, denominator = value.replace(",", "").split("/")
        result = Decimal(numerator.strip()) / Decimal(denominator.strip())
        return normalize_number(str(result)) if result.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def complete_answer(text: str, stop_reason: str) -> str | None:
    if stop_reason not in STOP_REASONS:
        raise ValueError("Missing/unknown generation stopping reason")
    candidates = []
    for match in TAG.finditer(text):
        # A budget cut at a trailing digit may be halfway through a multi-token number.
        if match.group(2) or stop_reason == "eos":
            candidates.append((match.start(), scalar(match.group(1))))
    candidates += [(match.start(), scalar(match.group(1))) for match in BOX.finditer(text)]
    return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def final_answer_stop(text: str) -> bool:
    return complete_answer(text, "completed_answer") is not None


def summarize_completed_math(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Unrun math is unavailable, not score zero")
    correct = completed = 0
    reasons = {reason: 0 for reason in STOP_REASONS}
    for row in rows:
        gold = numeric_answer(row["reference"], strict=True)
        if gold is None:
            raise ValueError("Invalid original GSM8K numeric reference")
        prediction = complete_answer(row["response"], row["stop_reason"])
        if prediction != row["prediction"] or row["correct"] is not (prediction == gold):
            raise ValueError("Substituted completion-aware math record")
        if prediction is not None:
            completed += 1
        correct += int(row["correct"])
        reasons[row["stop_reason"]] += 1
    return {
        "sample_count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "completed_answers": completed,
        "completed_fraction": completed / len(rows),
        "wilson_95": wilson(correct, len(rows)),
        "stopping_counts": reasons,
        "scoring": "completed #### numeric line or closed boxed scalar; no last-number fallback",
    }
