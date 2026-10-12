"""Real held-out GSM8K scoring and WikiText native-token/byte likelihood observations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from .data import collate
from .evaluation import predict
from .model import SurgicalChild
from .objectives import encode_text, math_prompt

NUMBER = r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"


def normalize_number(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", "").replace("$", "").replace("−", "-").strip())
        if not number.is_finite():
            return None
        return str(number.normalize()) if number else "0"
    except InvalidOperation:
        return None


def numeric_answer(text: str, *, strict: bool = False) -> str | None:
    tagged = re.findall(r"####\s*(" + NUMBER + r")", text)
    if tagged:
        return normalize_number(tagged[-1])
    if strict:
        return None
    boxed = re.findall(r"\\boxed\{\s*(" + NUMBER + r")\s*\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    numbers = re.findall(NUMBER, text)
    return normalize_number(numbers[-1]) if numbers else None


def wilson(successes: int, total: int) -> list[float] | None:
    if total < 1:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Invalid success count")
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def summarize_math(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Missing GSM8K observations cannot become a zero score")
    for row in rows:
        gold = numeric_answer(row["reference"], strict=True)
        if gold is None:
            raise ValueError("GSM8K reference has no valid #### numeric answer")
        prediction = numeric_answer(row["response"])
        if row["prediction"] != prediction or row["correct"] is not (prediction == gold):
            raise ValueError("Manually substituted/inconsistent GSM8K metric record")
    successes = sum(row["correct"] for row in rows)
    strict_successes = sum(
        numeric_answer(r["response"], strict=True) == numeric_answer(r["reference"], strict=True)
        for r in rows
    )
    return {
        "sample_count": len(rows),
        "correct": successes,
        "accuracy": successes / len(rows),
        "strict_accuracy": strict_successes / len(rows),
        "wilson_95": wilson(successes, len(rows)),
        "scoring": "numeric: last ####, boxed, or final occurrence; strict diagnostic",
    }


def summarize_language(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Missing language observations cannot become a zero score")
    if any(
        r["target_tokens"] < 1
        or r["text_bytes"] < 1
        or not math.isfinite(r["nll_sum"])
        or r["nll_sum"] < 0
        for r in rows
    ):
        raise ValueError("Invalid language-likelihood observations")
    nll = sum(r["nll_sum"] for r in rows)
    tokens, byte_count = sum(r["target_tokens"] for r in rows), sum(r["text_bytes"] for r in rows)
    return {
        "sample_count": len(rows),
        "target_tokens": tokens,
        "text_bytes": byte_count,
        "nll_sum": nll,
        "token_nll": nll / tokens,
        "bits_per_byte": nll / (byte_count * math.log(2)),
        "native_token_perplexity": math.exp(nll / tokens) if nll / tokens < 700 else None,
        "protocol": "text windows with BOS/EOS; not official word-level perplexity",
    }


@torch.inference_mode()
def generate_tokens(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    ablation: str = "none",
) -> tuple[str, list[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_input_tokens:
        raise ValueError(
            "Benchmark prompt exceeds registered input budget; never silently truncate"
        )
    if isinstance(model, SurgicalChild) and len(ids) + max_new_tokens > model.context_limit:
        raise ValueError("Benchmark generation exceeds the transplanted position budget")
    sequence = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = []
    model.eval()
    for _ in range(max_new_tokens):
        token = int(predict(model, {"input_ids": sequence}, ablation).logits[0, -1].argmax())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        sequence = torch.cat([sequence, torch.tensor([[token]], device=sequence.device)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated


def observe_math(
    model,
    tokenizer,
    examples: list[dict],
    target: str | Path,
    *,
    max_new_tokens: int = 128,
    max_input_tokens: int = 1024,
    ablation: str = "none",
) -> dict:
    path = Path(target)
    if path.exists():
        raise FileExistsError("Preserve previous benchmark records; use a new output file")
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = []
    for example in examples:
        started = time.monotonic()
        response, tokens = generate_tokens(
            model,
            tokenizer,
            math_prompt(example["question"]),
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            ablation=ablation,
        )
        gold = numeric_answer(example["answer"], strict=True)
        if gold is None:
            raise ValueError("Invalid published GSM8K reference")
        prediction = numeric_answer(response)
        row = {
            "id": example["source_index"],
            "split": "test",
            "question_sha256": hashlib.sha256(example["question"].encode()).hexdigest(),
            "reference": example["answer"],
            "response": response,
            "generated_ids": tokens,
            "prediction": prediction,
            "correct": prediction == gold,
            "elapsed_seconds": time.monotonic() - started,
        }
        observations.append(row)
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    return summarize_math(observations)


@torch.inference_mode()
def observe_language(
    model,
    tokenizer,
    chunks: list[str],
    target: str | Path,
    *,
    max_length: int = 256,
    ablation: str = "none",
) -> dict:
    path = Path(target)
    if path.exists():
        raise FileExistsError("Preserve previous language observations")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = encode_text(chunks, tokenizer, max_length)
    observations = []
    model.eval()
    for index, (text, record) in enumerate(zip(chunks, records, strict=True)):
        started = time.monotonic()
        batch = collate([record], tokenizer.pad_token_id, next(model.parameters()).device)
        output = predict(model, batch, ablation)
        count = sum(label != -100 for label in record.labels[1:])
        nll = float(output.loss) * count
        row = {
            "id": index,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_bytes": len(text.encode()),
            "target_tokens": count,
            "nll_sum": nll,
            "elapsed_seconds": time.monotonic() - started,
        }
        observations.append(row)
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return summarize_language(observations)
