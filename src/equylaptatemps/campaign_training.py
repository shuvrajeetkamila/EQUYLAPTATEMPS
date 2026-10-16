"""Resumable direct child loss on real data; safe NEW-tensor/optimizer checkpoints only."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .alignment_stats import gradient_summary
from .audit import frozen_fingerprint, sha256_file
from .data import collate
from .evaluation import evaluate_encoded


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_resumable(
    child,
    tokenizer,
    records,
    options,
    *,
    validation,
    output,
    contract,
    diagnostic_records=None,
    checkpoint_every=8,
    deadline=None,
    interrupt_after=None,
):
    """Resume only matching config/data/code/frozen semantics; no pickle or teacher tensors.

    Diagnostics may be a named subset, never quietly presented as full-pool loss.
    interrupt_after is an engineering-test/recovery limit, not a scientific metric.
    """
    options.validate()
    child.assert_trainable_contract()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = {n: p for n, p in child.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in names.items()}
    optimizer = torch.optim.AdamW(list(names.values()), lr=options.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(options.seed)
    fingerprint = frozen_fingerprint(child)
    datahash = stable_hash(
        [{"ids": r.ids, "labels": r.labels, "domain": r.domain} for r in records]
    )
    identity = {
        "contract": contract,
        "data_sha256": datahash,
        "frozen_sha256": fingerprint,
        "source_code_sha256": sha256_file(__file__),
        "steps": options.steps,
        "seed": options.seed,
        "learning_rate": options.learning_rate,
        "batch_size": options.batch_size,
        "gradient_checkpointing": options.gradient_checkpointing,
    }
    identityhash = stable_hash(identity)
    state_path = output / "state.json"
    tensor_path = output / "state.safetensors"
    step = 0
    order = []
    cursor = 0
    trace = []
    initial_metrics = None
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if (
            saved["identity_sha256"] != identityhash
            or sha256_file(tensor_path) != saved["tensors_sha256"]
        ):
            raise ValueError("Unsafe training resume: config/data/code/frozen/checkpoint mismatch")
        tensors = load_file(str(tensor_path))
        with torch.no_grad():
            for n, p in names.items():
                p.copy_(tensors["parameter/" + n])
                initial[n] = tensors["initial/" + n]
                if "avg/" + n in tensors:
                    optimizer.state[p] = {
                        "step": tensors["step/" + n],
                        "exp_avg": tensors["avg/" + n].to(p.device),
                        "exp_avg_sq": tensors["square/" + n].to(p.device),
                    }
        generator.set_state(tensors["rng"])
        step = saved["completed_steps"]
        order = saved["order"]
        cursor = saved["cursor"]
        trace = saved["trace"]
        initial_metrics = saved["initial_metrics"]
    diagnostic_records = diagnostic_records or records
    if initial_metrics is None:
        initial_metrics = evaluate_encoded(
            child, diagnostic_records, tokenizer, batch_size=options.batch_size
        )
    started = time.monotonic()
    child.gradient_checkpointing = options.gradient_checkpointing

    def checkpoint():
        tensors = {"rng": generator.get_state()}
        for n, p in names.items():
            tensors["parameter/" + n] = p.detach().cpu().contiguous()
            tensors["initial/" + n] = initial[n].contiguous()
            if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                state = optimizer.state[p]
                tensors["avg/" + n] = state["exp_avg"].cpu().contiguous()
                tensors["square/" + n] = state["exp_avg_sq"].cpu().contiguous()
                tensors["step/" + n] = state["step"].cpu().contiguous()
        temporary = output / "state.new.safetensors"
        save_file(tensors, str(temporary))
        os.replace(temporary, tensor_path)
        record = {
            "identity": identity,
            "identity_sha256": identityhash,
            "tensors_sha256": sha256_file(tensor_path),
            "completed_steps": step,
            "order": order,
            "cursor": cursor,
            "trace": trace,
            "initial_metrics": initial_metrics,
        }
        temp = output / "state.new.json"
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(temp, state_path)

    child.train()
    try:
        while step < options.steps:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if interrupt_after is not None and step >= interrupt_after:
                break
            if cursor >= len(order):
                order = torch.randperm(len(records), generator=generator).tolist()
                cursor = 0
            chosen = order[cursor : cursor + options.batch_size]
            cursor += len(chosen)
            batch = collate([records[i] for i in chosen], tokenizer.pad_token_id, child.device)
            optimizer.zero_grad(set_to_none=True)
            loss = child(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite direct child calibration loss")
            loss.backward()
            gradients = gradient_summary(child)
            norm = float(
                torch.nn.utils.clip_grad_norm_(list(names.values()), 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            step += 1
            event = {
                "step": step,
                "record_indices": chosen,
                "batch_token_nll": float(loss.detach()),
                "gradient_norm_before_clip": norm,
                "gradients": gradients,
                "elapsed_session_seconds": time.monotonic() - started,
            }
            trace.append(event)
            with (output / "journal.jsonl").open("a") as f:
                f.write(json.dumps(event, allow_nan=False) + "\n")
            if step % checkpoint_every == 0:
                checkpoint()
    except Exception:
        checkpoint()
        raise
    finally:
        child.eval()
    checkpoint()
    if frozen_fingerprint(child) != fingerprint:
        raise ValueError("Frozen original/random sector changed")
    complete = step == options.steps
    report = {
        "execution_status": "COMPLETED" if complete else "INTERRUPTED",
        "completed_steps": step,
        "identity_sha256": identityhash,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "objective": "direct child real-text/human-ground-truth token loss; no teachers",
        "initial_train_diagnostic": initial_metrics,
        "diagnostic_examples": len(diagnostic_records),
        "training_pool_examples": len(records),
        "final_train_diagnostic": evaluate_encoded(
            child, diagnostic_records, tokenizer, batch_size=options.batch_size
        )
        if complete
        else None,
        "validation": evaluate_encoded(child, validation, tokenizer, batch_size=options.batch_size)
        if complete
        else None,
        "trace": trace,
        "new_parameter_updates": {
            n: {
                "l2_update": float((p.detach().cpu() - initial[n]).norm()),
                "relative_update": float(
                    (p.detach().cpu() - initial[n]).norm() / (initial[n].norm() + 1e-12)
                ),
            }
            for n, p in names.items()
        },
        "parent_weights_unchanged": True,
        "random_sector_control": bool(getattr(child, "random_graft_control", False)),
        "checkpoint_sha256": sha256_file(tensor_path),
        "trainable_parameters": sum(p.numel() for p in names.values()),
        "session_runtime_seconds": time.monotonic() - started,
    }
    (output / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
