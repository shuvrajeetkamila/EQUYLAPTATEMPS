"""Offline architecture/schema checks only; fixtures are not pretrained-capability evidence."""

import json
from pathlib import Path

import pytest
import torch

from equylaptatemps.audit import frozen_fingerprint
from equylaptatemps.benchdata import gsm_splits, text_chunks
from equylaptatemps.objectives import encode_math, encode_text
from equylaptatemps.parents import LoadedParents
from equylaptatemps.recipe import LayerSpan
from equylaptatemps.references import load_reference, save_reference
from equylaptatemps.sectors import (
    activation_statistics,
    identity_sector,
    sector_inventory,
    trace_child,
)
from equylaptatemps.training import TrainingOptions, fit_encoded_seams


def test_real_text_objective_has_no_teacher_and_masks_only_initial_token(child_bundle):
    child, tokenizer, *_ = child_bundle
    rows = encode_text(["Real text fixture."], tokenizer, 64)
    assert rows[0].labels[0] == -100
    assert rows[0].labels[1:] == rows[0].ids[1:]
    frozen = frozen_fingerprint(child)
    result = fit_encoded_seams(child, tokenizer, rows, TrainingOptions(steps=1, max_length=64))
    assert result["parent_kind"] == "synthetic"
    assert frozen_fingerprint(child) == frozen
    assert len(result["trainable_tensors"]) == 8


def test_math_human_reference_schema_and_no_silent_truncation(child_bundle):
    _, tokenizer, *_ = child_bundle
    rows = [{"question": "One plus one?", "answer": "Two. #### 2", "source_index": 0}]
    encoded, omitted = encode_math(rows, tokenizer, 256)
    assert encoded[0].domain == "math" and not omitted
    with pytest.raises(ValueError, match="context budget"):
        encode_math(rows, tokenizer, 8)
    filtered, omitted = encode_math(rows, tokenizer, 8, filter_oversize=True)
    assert not filtered and omitted == [0]


def test_calibration_journal_records_every_step_and_refuses_overwrite(child_bundle, tmp_path):
    child, tokenizer, *_ = child_bundle
    rows = encode_text(["A real-text schema fixture."], tokenizer, 64)
    log = tmp_path / "steps.jsonl"
    fit_encoded_seams(child, tokenizer, rows, TrainingOptions(steps=2, max_length=64), journal=log)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [event["step"] for event in events if event["event"] == "step"] == [1, 2]
    assert events[-1]["frozen_tensors_unchanged"]
    with pytest.raises(FileExistsError):
        fit_encoded_seams(child, tokenizer, rows, TrainingOptions(steps=1), journal=log)


def test_complete_sector_inventory_and_actual_execution_trace(child_bundle):
    child, _, _, recipient, _ = child_bundle
    inventory = sector_inventory(recipient, child.recipe.recipient, [LayerSpan(1, 3)])
    assert inventory["parent_kind"] == "synthetic"
    sector = inventory["sectors"][0]
    assert sector["block_boundaries"] == [1, 3]
    assert sector["parameter_count"] == sum(
        p.numel() for block in recipient.backbone.layers[1:3] for p in block.parameters()
    )
    assert sector["baseline_performance"] is None and sector["selection_status"] == "NOT_RUN"
    with torch.inference_mode(), trace_child(child) as calls:
        child(torch.tensor([[8, 9, 10]]))
    assert calls == {"prefix": [1], "graft": [1, 1], "suffix": [1]}


def test_identity_ablation_restores_hooks_and_preserves_weights(child_bundle):
    _, _, _, parent, _ = child_bundle
    ids = torch.tensor([[8, 9, 10]])
    before = parent(ids, use_cache=False).logits.detach()
    with identity_sector(parent, LayerSpan(1, 3)):
        ablated = parent(ids, use_cache=False).logits.detach()
    after = parent(ids, use_cache=False).logits.detach()
    assert not torch.equal(before, ablated)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_activation_statistics_are_observations_not_training_targets(child_bundle):
    _, _, _, parent, _ = child_bundle
    result = activation_statistics(parent, [{"input_ids": torch.tensor([[8, 9, 10]])}])
    assert result["kind"] == "measured"
    assert all(row["calls"] == 1 and row["nonfinite"] == 0 for row in result["blocks"])
    assert result["blocks"][0]["input_rms"] > 0
    assert all(
        not isinstance(value, torch.Tensor) for row in result["blocks"] for value in row.values()
    )


def test_bridge_only_reference_contains_no_parent_tensors_and_reloads(child_bundle, tmp_path):
    child, tokenizer, provenance, recipient, donor = child_bundle
    saved = save_reference(child, tmp_path / "reference", provenance)
    assert saved["self_contained"] is False and saved["parent_kind"] == "synthetic"
    parents = LoadedParents(recipient, donor, tokenizer, provenance)
    restored = load_reference(tmp_path / "reference", parents)
    ids = torch.tensor([[8, 9, 10]])
    with torch.inference_mode():
        torch.testing.assert_close(child(ids).logits, restored(ids).logits, rtol=0, atol=0)
    assert (tmp_path / "reference/interfaces.safetensors").stat().st_size < 10000


def test_reference_detects_changed_bridge_checksum(child_bundle, tmp_path):
    child, tokenizer, provenance, a, b = child_bundle
    save_reference(child, tmp_path / "reference", provenance)
    path = tmp_path / "reference/interfaces.safetensors"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_reference(tmp_path / "reference", LoadedParents(a, b, tokenizer, provenance))


def test_benchmark_train_dev_separation_and_text_windows():
    rows = [{"question": str(i), "answer": f"#### {i}"} for i in range(10)]
    train, dev = gsm_splits(rows, seed=42, train_count=4, dev_count=2)
    assert {r["source_index"] for r in train}.isdisjoint({r["source_index"] for r in dev})
    assert (train, dev) == gsm_splits(rows, seed=42, train_count=4, dev_count=2)
    assert text_chunks("abcdefghijklmnop" * 3, characters=16, limit=2) == ["abcdefghijklmnop"] * 2
    with pytest.raises(ValueError):
        gsm_splits(rows, seed=42, train_count=9, dev_count=2)


def test_published_synthetic_checkpoint_was_not_replaced():
    root = Path(__file__).resolve().parents[1]
    archive = root / "examples/equylaptatemps-synthetic-child-v0.1.0.zip"
    assert archive.stat().st_size == 88457
