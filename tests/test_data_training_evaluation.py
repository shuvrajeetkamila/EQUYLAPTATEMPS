import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from equylaptatemps.audit import frozen_fingerprint
from equylaptatemps.data import Record, collate, encode_records, ensure_disjoint, read_records
from equylaptatemps.evaluation import evaluate_answers, evaluate_encoded, greedy_answer
from equylaptatemps.tokenization import (
    ByteTokenizer,
    check_tokenizer,
    prompt_ids,
    tokenizer_fingerprint,
)
from equylaptatemps.training import TrainingOptions, fit_seams


@pytest.mark.parametrize("text", ["hello", "π ∑ 数学", "", "a\nline"])
def test_byte_codec_roundtrips_and_is_independent(text):
    a, b = ByteTokenizer(), ByteTokenizer(37)
    assert a.decode(a.encode(text)) == b.decode(b.encode(text)) == text
    assert tokenizer_fingerprint(a) != tokenizer_fingerprint(b)
    check_tokenizer(a, 259)
    with pytest.raises(ValueError, match="frozen vocabulary"):
        check_tokenizer(a, 258)


def test_response_masks_match_inference_prefix():
    tokenizer = ByteTokenizer()
    record = Record("1+1=", "2", "math")
    row = encode_records([record], tokenizer, 64)[0]
    prefix = prompt_ids(tokenizer, record.prompt)
    assert row.ids[: len(prefix)] == prefix
    assert row.labels[: len(prefix)] == [-100] * len(prefix)
    assert row.labels[len(prefix) :] == tokenizer.encode(" 2") + [tokenizer.eos_token_id]
    batch = collate([row, encode_records([Record("x", "y")], tokenizer, 64)[0]], 0)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)


def test_answers_are_never_silently_truncated():
    with pytest.raises(ValueError, match="never silently truncated"):
        encode_records([Record("A long prompt", "a long answer")], ByteTokenizer(), 8)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "not json",
        "{}\n",
        '{"prompt":"x", "answer":""}\n',
        '{"prompt":"x", "answer":"y", "teacher_logits":[1,2]}\n',
        '{"prompt":5, "answer":"y"}\n',
    ],
)
def test_bad_supervision_rejected(tmp_path, contents):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_records(path)


def test_jsonl_load_and_prompt_leakage_detection(tmp_path):
    path = tmp_path / "good.jsonl"
    path.write_text(json.dumps({"prompt": "1+1=", "answer": "2", "domain": "math"}) + "\n")
    assert read_records(path) == [Record("1+1=", "2", "math")]
    with pytest.raises(ValueError, match="overlapping prompts"):
        ensure_disjoint([Record("What IS x?", "1")], [Record(" what is  x? ", "1")])


def test_evaluation_is_token_weighted_and_grouped(child_bundle):
    child, tokenizer, *_ = child_bundle
    records = encode_records([Record("x", "a", "math"), Record("y", "abcd", "code")], tokenizer, 64)
    batch = collate(records, 0)
    with torch.inference_mode():
        logits = child(**batch).logits
        targets = batch["labels"][:, 1:]
        total = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), targets, ignore_index=-100, reduction="sum"
        )
        count = int((targets != -100).sum())
    child.train()
    result = evaluate_encoded(child, records, tokenizer, batch_size=1)
    assert child.training is True
    assert result["groups"]["overall"]["token_nll"] == pytest.approx(float(total) / count, abs=1e-6)
    assert result["groups"]["overall"]["target_tokens"] == count
    assert set(result["groups"]) == {"overall", "domain:math", "domain:code"}
    batched = evaluate_encoded(child, records, tokenizer, batch_size=2)
    assert batched["groups"]["overall"]["token_nll"] == pytest.approx(
        result["groups"]["overall"]["token_nll"], abs=1e-6
    )


def test_calibration_improves_smoke_loss_without_changing_any_parent_tensor(child_bundle):
    child, tokenizer, *_ = child_bundle
    frozen = frozen_fingerprint(child)
    records = [Record("1+1=", "2"), Record("2+2=", "4")]
    report = fit_seams(
        child,
        tokenizer,
        records,
        TrainingOptions(steps=10, batch_size=2, learning_rate=0.02, max_length=64),
    )
    assert (
        report["final_train"]["groups"]["overall"]["token_nll"]
        < (report["initial_train"]["groups"]["overall"]["token_nll"])
    )
    assert report["frozen_tensors_unchanged"]
    assert frozen_fingerprint(child) == frozen
    assert report["capability_claim"].startswith("none")


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": True},
        {"batch_size": 0},
        {"max_length": 1},
        {"learning_rate": float("nan")},
        {"learning_rate": -0.1},
        {"learning_rate": float("inf")},
        {"seed": -1},
        {"gradient_checkpointing": "yes"},
    ],
)
def test_invalid_training_options_rejected(changes):
    with pytest.raises(ValueError):
        replace(TrainingOptions(), **changes).validate()


def test_calibration_detects_heldout_leakage(child_bundle):
    child, tokenizer, *_ = child_bundle
    rows = [Record("1+1=", "2")]
    with pytest.raises(ValueError, match="leakage"):
        fit_seams(child, tokenizer, rows, TrainingOptions(), validation=rows)


def test_generation_and_strict_answer_metrics_are_bounded(child_bundle):
    child, tokenizer, *_ = child_bundle
    child.train()
    answer = greedy_answer(child, tokenizer, "1+1=", max_new_tokens=2)
    assert isinstance(answer, str) and child.training
    report = evaluate_answers(child, tokenizer, [Record("1+1=", "2")], max_new_tokens=2)
    assert 0 <= report["exact_match"] <= 1
    assert "not semantic" in report["warning"]
    with pytest.raises(ValueError, match="budget"):
        greedy_answer(child, tokenizer, "x" * 250, max_new_tokens=10)


def test_bfloat16_calibration_preserves_frozen_tensors(child_bundle):
    child, tokenizer, *_ = child_bundle
    child.to(torch.bfloat16)
    frozen = frozen_fingerprint(child)
    report = fit_seams(
        child,
        tokenizer,
        [Record("1+1=", "2")],
        TrainingOptions(steps=2, learning_rate=0.02, max_length=64, gradient_checkpointing=True),
    )
    assert report["frozen_tensors_unchanged"]
    assert frozen_fingerprint(child) == frozen


def test_empty_answers_and_domains_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        Record("x", "")
    with pytest.raises(ValueError, match="nonempty"):
        Record("x", "y", "")
