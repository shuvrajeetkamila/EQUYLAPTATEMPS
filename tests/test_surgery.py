from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from equylaptatemps.audit import frozen_fingerprint, module_fingerprint
from equylaptatemps.backbones import StackSegment, config_from_dict, extract_parent
from equylaptatemps.data import Record, collate, encode_records
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan
from equylaptatemps.synthetic import demo_recipe, tiny_parent
from equylaptatemps.tokenization import ByteTokenizer


@pytest.mark.parametrize("family", ["mamba", "qwen2"])
def test_segment_executor_matches_native_parent(family):
    parent = tiny_parent(family, 11)
    parts = extract_parent(parent)
    config = config_from_dict(parent.config.to_dict())
    segment = StackSegment(config, parts.layers).eval()
    ids = torch.tensor([[8, 9, 10, 11], [12, 13, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with torch.inference_mode():
        native = parent(ids, attention_mask=attention, use_cache=False).logits.float()
        hidden = segment(parts.embedding(ids), attention)
        surgery = parts.head(parts.norm(hidden).to(parts.head.weight.dtype)).float()
    torch.testing.assert_close(
        surgery[attention.bool()], native[attention.bool()], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_entire_blocks_transplanted_without_weight_arithmetic(reverse):
    recipe = demo_recipe(reverse=reverse)
    a = tiny_parent(recipe.recipient.identity.family, 11)
    b = tiny_parent(recipe.donor.identity.family, 22)
    donor_parts = extract_parent(b)
    source = torch.nn.ModuleList(donor_parts.layers[1:3])
    original_hash = module_fingerprint(source)
    child = child_from_parents(recipe, a, b)
    assert module_fingerprint(child.graft.layers) == original_hash
    assert child.graft.layers[0] is donor_parts.layers[1]
    assert child.graft.layers[1] is donor_parts.layers[2]
    retained_ids = {id(p) for p in child.parameters()}
    for removed in extract_parent(a).layers[1:3]:
        assert not retained_ids & {id(p) for p in removed.parameters()}
    assert id(donor_parts.embedding.weight) not in retained_ids
    assert id(donor_parts.head.weight) not in retained_ids
    assert not any(name in {"recipient", "donor"} for name, _ in child.named_modules())
    assert child.structure_report()["production_bypass"] is False
    child.assert_trainable_contract()


@pytest.mark.parametrize("reverse", [False, True])
def test_every_donor_block_is_executed(reverse):
    recipe = demo_recipe(reverse=reverse)
    a = tiny_parent(recipe.recipient.identity.family, 11)
    b = tiny_parent(recipe.donor.identity.family, 22)
    child = child_from_parents(recipe, a, b)
    calls = []
    hooks = [
        layer.register_forward_hook(lambda module, args, output: calls.append(id(module)))
        for layer in child.graft.layers
    ]
    try:
        child(torch.tensor([[8, 9, 10]]))
        assert calls == [id(layer) for layer in child.graft.layers]
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("reverse", [False, True])
def test_future_tokens_cannot_change_prefix_logits(reverse):
    recipe = demo_recipe(reverse=reverse)
    child = child_from_parents(
        recipe,
        tiny_parent(recipe.recipient.identity.family, 11),
        tiny_parent(recipe.donor.identity.family, 22),
    )
    with torch.inference_mode():
        a = child(torch.tensor([[8, 9, 10, 11, 12]])).logits
        b = child(torch.tensor([[8, 9, 10, 24, 25]])).logits
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.equal(a[:, -1], b[:, -1])


def test_padding_does_not_change_real_token_logits(child_bundle):
    child, *_ = child_bundle
    with torch.inference_mode():
        unpadded = child(torch.tensor([[8, 9, 10]])).logits
        padded = child(
            torch.tensor([[8, 9, 10, 0, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0, 0]])
        ).logits
    torch.testing.assert_close(unpadded, padded[:, :3], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("reverse", [False, True])
def test_checkpointed_gradients_equal_plain_gradients_through_frozen_blocks(reverse):
    recipe = demo_recipe(reverse=reverse)
    a = tiny_parent(recipe.recipient.identity.family, 11)
    b = tiny_parent(recipe.donor.identity.family, 22)
    child = child_from_parents(recipe, a, b)
    records = encode_records([Record("1+1=", "2"), Record("2+2=", "4")], ByteTokenizer(), 64)
    batch = collate(records, 0)
    frozen_before = frozen_fingerprint(child)
    child.train()
    child.gradient_checkpointing = False
    child(**batch).loss.backward()
    expected = [p.grad.clone() for p in child.seam_parameters()]
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in expected)
    assert all(
        p.grad is None
        for name, p in child.named_parameters()
        if not name.startswith(("input_seam.", "output_seam."))
    )
    child.zero_grad(set_to_none=True)
    child.gradient_checkpointing = True
    child(**batch).loss.backward()
    for parameter, gradient in zip(child.seam_parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-6, rtol=1e-6)
    torch.optim.AdamW(child.seam_parameters(), lr=0.01).step()
    assert frozen_fingerprint(child) == frozen_before
    assert child.training and child.input_seam.training
    assert not child.graft.training and not child.suffix.training


def test_loss_is_shifted_and_masks_prompts_and_padding(child_bundle):
    child, tokenizer, *_ = child_bundle
    batch = collate(encode_records([Record("x", "y"), Record("longer", "ab")], tokenizer, 64), 0)
    result = child(**batch)
    expected = F.cross_entropy(
        result.logits[:, :-1].reshape(-1, 259),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(result.loss, expected)


@pytest.mark.parametrize("attention", [[[0, 1, 1]], [[1, 0, 1]], [[0, 0, 0]], [[1, 2, 0]]])
def test_invalid_padding_rejected(child_bundle, attention):
    child, *_ = child_bundle
    with pytest.raises(ValueError, match="padded|binary"):
        child(torch.tensor([[8, 9, 10]]), attention_mask=torch.tensor(attention))


def test_invalid_inputs_and_labels_rejected(child_bundle):
    child, *_ = child_bundle
    with pytest.raises(ValueError, match="recipient vocabulary"):
        child(torch.tensor([[300, 301]]))
    with pytest.raises(ValueError, match="int64"):
        child(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="context limit"):
        child(torch.ones(1, 257, dtype=torch.long))
    with pytest.raises(ValueError, match="supervised"):
        child(torch.tensor([[8, 9]]), labels=torch.tensor([[-100, -100]]))
    with pytest.raises(ValueError, match="Padding"):
        child(
            torch.tensor([[8, 0]]),
            attention_mask=torch.tensor([[1, 0]]),
            labels=torch.tensor([[-100, 1]]),
        )
    with pytest.raises(ValueError, match="evaluation-only"):
        child.train()(torch.tensor([[8, 9]]), ablation="bypass_graft")


def test_different_depths_and_full_recipient_cut_are_supported(child_bundle):
    _, _, _, a, b = child_bundle
    recipe = replace(demo_recipe(), recipient_cut=LayerSpan(0, 4), donor_sector=LayerSpan(0, 1))
    child = child_from_parents(recipe, a, b)
    assert len(child.prefix.layers) == len(child.suffix.layers) == 0
    assert len(child.graft.layers) == 1
    assert child(torch.tensor([[8, 9]])).logits.shape == (1, 2, 259)


def test_mixed_parent_precision_rejected(child_bundle):
    _, _, _, a, b = child_bundle
    b.to(torch.bfloat16)
    with pytest.raises(ValueError, match="common floating-point dtype"):
        child_from_parents(demo_recipe(), a, b)


def test_rotary_frequency_precision_survives_child_dtype_move(child_bundle):
    child, *_ = child_bundle
    frequencies = child.graft.rotary.inv_freq.clone()
    child.to(torch.bfloat16)
    assert child.graft.rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(child.graft.rotary.inv_freq, frequencies, atol=0, rtol=0)
