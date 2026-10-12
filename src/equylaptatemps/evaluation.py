"""Token-weighted held-out metrics, native-parent comparisons, and graft necessity ablations."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .data import EncodedRecord, Record, collate, encode_records
from .model import SurgicalChild
from .tokenization import Tokenizer, prompt_ids


def predict(model, batch: dict, ablation: str = "none"):
    if isinstance(model, SurgicalChild):
        return model(**batch, ablation=ablation)
    if ablation != "none":
        raise ValueError("Graft ablations apply only to a surgical child")
    return model(**batch, use_cache=False)


@torch.inference_mode()
def evaluate_encoded(
    model,
    encoded: list[EncodedRecord],
    tokenizer: Tokenizer,
    *,
    batch_size: int = 1,
    ablation: str = "none",
) -> dict:
    if batch_size < 1 or not encoded:
        raise ValueError("Evaluation needs a positive batch size and nonempty dataset")
    was_training = model.training
    model.eval()
    groups: dict[str, dict] = {}
    try:
        for offset in range(0, len(encoded), batch_size):
            rows = encoded[offset : offset + batch_size]
            batch = collate(rows, tokenizer.pad_token_id, next(model.parameters()).device)
            output = predict(model, batch, ablation)
            targets = batch["labels"][:, 1:]
            losses = F.cross_entropy(
                output.logits[:, :-1].float().transpose(1, 2),
                targets,
                ignore_index=-100,
                reduction="none",
            )
            for index, record in enumerate(rows):
                count = int((targets[index] != -100).sum())
                nll = float(losses[index].sum())
                if not math.isfinite(nll):
                    raise FloatingPointError("Nonfinite evaluation loss")
                for key in ("overall", f"domain:{record.domain}"):
                    group = groups.setdefault(
                        key, {"nll_sum": 0.0, "target_tokens": 0, "examples": 0}
                    )
                    group["nll_sum"] += nll
                    group["target_tokens"] += count
                    group["examples"] += 1
        for group in groups.values():
            group["token_nll"] = group["nll_sum"] / group["target_tokens"]
            group["perplexity"] = math.exp(group["token_nll"]) if group["token_nll"] < 700 else None
        return {
            "ablation": ablation,
            "groups": groups,
            "metric": "recipient-token response NLL; not proof of task mastery",
        }
    finally:
        model.train(was_training)


@torch.inference_mode()
def greedy_answer(
    model, tokenizer: Tokenizer, prompt: str, *, max_new_tokens: int = 64, ablation: str = "none"
) -> str:
    if max_new_tokens < 1 or not prompt.strip():
        raise ValueError("Generation requires a nonempty prompt and positive max_new_tokens")
    ids = prompt_ids(tokenizer, prompt)
    if (
        isinstance(model, SurgicalChild)
        and model.context_limit is not None
        and len(ids) + max_new_tokens > model.context_limit
    ):
        raise ValueError("Prompt plus generation budget exceeds the transplanted context limit")
    device = next(model.parameters()).device
    sequence = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            output = predict(model, {"input_ids": sequence}, ablation)
            token = int(output.logits[0, -1].argmax())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            sequence = torch.cat([sequence, torch.tensor([[token]], device=device)], dim=1)
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    finally:
        model.train(was_training)


def evaluate_answers(
    model,
    tokenizer: Tokenizer,
    records: list[Record],
    *,
    max_new_tokens: int = 64,
    ablation: str = "none",
) -> dict:
    if not records:
        raise ValueError("Answer evaluation needs a nonempty dataset")
    results = []
    for record in records:
        prediction = greedy_answer(
            model, tokenizer, record.prompt, max_new_tokens=max_new_tokens, ablation=ablation
        )
        results.append(
            {
                "prompt": record.prompt,
                "reference": record.answer,
                "prediction": prediction,
                "domain": record.domain,
                "exact_match": prediction == record.answer.strip(),
            }
        )
    domains = sorted({record.domain for record in records})
    return {
        "exact_match": sum(result["exact_match"] for result in results) / len(results),
        "by_domain": {
            domain: sum(r["exact_match"] for r in results if r["domain"] == domain)
            / sum(r["domain"] == domain for r in results)
            for domain in domains
        },
        "results": results,
        "warning": "Strict string equality is a smoke metric, not semantic/code correctness.",
    }


def evaluate_records(model, tokenizer, records, *, max_length=256, batch_size=1, ablation="none"):
    return evaluate_encoded(
        model,
        encode_records(records, tokenizer, max_length),
        tokenizer,
        batch_size=batch_size,
        ablation=ablation,
    )
