"""Ground-truth supervision only. No teacher model, parent logits, or synthetic parent answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenization import Tokenizer, prompt_ids


@dataclass(frozen=True)
class Record:
    prompt: str
    answer: str
    domain: str = "general"

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.prompt, self.answer, self.domain)
        ):
            raise ValueError("Record prompt, answer, and domain must be nonempty strings")


@dataclass
class EncodedRecord:
    ids: list[int]
    labels: list[int]
    domain: str


def read_records(path: str | Path) -> list[Record]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not {"prompt", "answer"} <= set(item):
                    raise ValueError("Required fields: prompt, answer; optional field: domain")
                if set(item) - {"prompt", "answer", "domain"}:
                    raise ValueError("Unknown fields; teacher outputs/logits are not accepted")
                if any(not isinstance(value, str) or not value.strip() for value in item.values()):
                    raise ValueError("All record values must be nonempty strings")
                records.append(Record(**item))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"No ground-truth examples in {path}")
    return records


def ensure_disjoint(train: list[Record], heldout: list[Record]) -> None:
    def canonical(record):
        return " ".join(record.prompt.split()).casefold()

    overlap = {canonical(r) for r in train} & {canonical(r) for r in heldout}
    if overlap:
        raise ValueError("Training and held-out sets have overlapping prompts; refusing leakage")


def encode_records(
    records: list[Record], tokenizer: Tokenizer, max_length: int
) -> list[EncodedRecord]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    encoded = []
    for index, record in enumerate(records):
        prefix = prompt_ids(tokenizer, record.prompt)
        if not prefix:
            raise ValueError("Tokenizer produced an empty inference prefix")
        answer = tokenizer.encode(" " + record.answer, add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        ids = prefix + answer
        if len(ids) > max_length:
            raise ValueError(
                f"Example {index} needs {len(ids)} tokens, exceeding max_length={max_length}; "
                "raise the limit or curate the data. Answers are never silently truncated."
            )
        encoded.append(EncodedRecord(ids, [-100] * len(prefix) + answer, record.domain))
    if not encoded:
        raise ValueError("An empty dataset cannot calibrate or evaluate seams")
    return encoded


def collate(records: list[EncodedRecord], pad_id: int, device: str | torch.device = "cpu") -> dict:
    if not records:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(record.ids) for record in records)
    ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for row, record in enumerate(records):
        length = len(record.ids)
        ids[row, :length] = torch.tensor(record.ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(record.labels, dtype=torch.long)
        attention[row, :length] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": attention.to(device),
        "labels": labels.to(device),
    }
