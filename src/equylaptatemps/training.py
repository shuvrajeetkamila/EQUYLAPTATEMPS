"""Calibrate new dense boundaries against human-provided ground truth, through frozen sectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .audit import frozen_fingerprint, module_fingerprint
from .data import Record, collate, encode_records, ensure_disjoint
from .evaluation import evaluate_encoded
from .model import SurgicalChild
from .tokenization import Tokenizer


@dataclass(frozen=True)
class TrainingOptions:
    steps: int = 100
    batch_size: int = 2
    learning_rate: float = 3e-4
    max_length: int = 256
    seed: int = 42
    gradient_checkpointing: bool = False

    def validate(self):
        if any(type(value) is not int for value in (self.steps, self.batch_size, self.max_length)):
            raise ValueError("steps, batch_size, and max_length must be integers")
        if self.steps < 1 or self.batch_size < 1 or self.max_length < 2:
            raise ValueError("steps, batch_size, and max_length must be positive (length >= 2)")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning_rate must be positive and finite")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Calibration seed must be an integer in [0, 2**32)")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")


def fit_seams(
    child: SurgicalChild,
    tokenizer: Tokenizer,
    records: list[Record],
    options: TrainingOptions,
    *,
    validation: list[Record] | None = None,
) -> dict:
    options.validate()
    child.assert_trainable_contract()
    if validation is not None:
        ensure_disjoint(records, validation)
    encoded = encode_records(records, tokenizer, options.max_length)
    heldout = (
        encode_records(validation, tokenizer, options.max_length)
        if validation is not None
        else None
    )
    frozen_before = frozen_fingerprint(child)
    graft_before = module_fingerprint(child.graft.layers)
    initial = evaluate_encoded(child, encoded, tokenizer, batch_size=options.batch_size)
    optimizer = torch.optim.AdamW(child.seam_parameters(), lr=options.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(options.seed)
    order = []
    cursor = 0
    losses = []
    child.gradient_checkpointing = options.gradient_checkpointing
    child.train()
    try:
        for step in range(options.steps):
            if cursor >= len(order):
                order = torch.randperm(len(encoded), generator=generator).tolist()
                cursor = 0
            selection = order[cursor : cursor + options.batch_size]
            cursor += len(selection)
            batch = collate([encoded[i] for i in selection], tokenizer.pad_token_id, child.device)
            optimizer.zero_grad(set_to_none=True)
            loss = child(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite calibration loss at step {step}")
            loss.backward()
            if not all(p.grad is not None for p in child.seam_parameters()):
                raise RuntimeError("A seam did not receive gradients through the frozen sector")
            torch.nn.utils.clip_grad_norm_(child.seam_parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            # A bounded trace keeps reports small even for long calibration runs.
            if (
                step == 0
                or step == options.steps - 1
                or (step + 1) % max(1, options.steps // 20) == 0
            ):
                losses.append({"step": step + 1, "batch_token_nll": float(loss.detach())})
    finally:
        child.eval()
    if (
        frozen_fingerprint(child) != frozen_before
        or module_fingerprint(child.graft.layers) != graft_before
    ):
        raise RuntimeError("Calibration changed a pretrained tensor; export refused")
    final = evaluate_encoded(child, encoded, tokenizer, batch_size=options.batch_size)
    validation_metrics = (
        None
        if heldout is None
        else evaluate_encoded(child, heldout, tokenizer, batch_size=options.batch_size)
    )
    return {
        "objective": "ground-truth response-token cross entropy",
        "optimizer_scope": "two newly initialized full-rank dense seams only",
        "steps": options.steps,
        "batch_size": options.batch_size,
        "learning_rate": options.learning_rate,
        "max_length": options.max_length,
        "seed": options.seed,
        "gradient_checkpointing": options.gradient_checkpointing,
        "initial_train": initial,
        "final_train": final,
        "validation": validation_metrics,
        "frozen_tensors_unchanged": True,
        "donor_sector_sha256": graft_before,
        "frozen_tensors_sha256": frozen_before,
        "trace": losses,
        "capability_claim": "none; calibration loss is not a capability benchmark",
    }
