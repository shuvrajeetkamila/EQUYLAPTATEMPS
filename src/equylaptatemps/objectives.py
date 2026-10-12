"""Human GSM8K train answers and real WikiText next-token supervision, never parent outputs."""

from __future__ import annotations

from .data import EncodedRecord


def math_prompt(question: str) -> str:
    return (
        "Question: " + question.strip() + "\n"
        "Answer: Let's think step by step. Finish with #### followed by the final number.\n"
    )


def encode_math(rows: list[dict], tokenizer, max_length: int, *, filter_oversize: bool = False):
    encoded, omitted = [], []
    for row in rows:
        prefix = tokenizer.encode(math_prompt(row["question"]), add_special_tokens=False)
        answer = tokenizer.encode(row["answer"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        ids = prefix + answer
        if len(ids) > max_length:
            if not filter_oversize:
                raise ValueError(
                    "GSM8K calibration reference exceeds the registered context budget"
                )
            omitted.append(row.get("source_index"))
            continue
        encoded.append(EncodedRecord(ids, [-100] * len(prefix) + answer, "math"))
    return encoded, omitted


def encode_text(chunks: list[str], tokenizer, max_length: int):
    encoded = []
    for text in chunks:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        bos = tokenizer.bos_token_id
        bos = tokenizer.eos_token_id if bos is None else bos
        ids = [bos, *tokens, tokenizer.eos_token_id]
        if not tokens or len(ids) > max_length:
            raise ValueError(
                "Real-text chunk exceeds max_length; change the registered chunk budget"
            )
        encoded.append(EncodedRecord(ids, [-100, *tokens, tokenizer.eos_token_id], "language"))
    if not encoded:
        raise ValueError("No real-text calibration samples")
    return encoded
