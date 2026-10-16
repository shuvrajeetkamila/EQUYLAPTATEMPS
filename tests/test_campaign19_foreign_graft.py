"""Campaign 19 foreign-graft machinery: offline synthetic tests only.

No pretrained weights and no experimental conditions are run here.
"""

import copy

import pytest
import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.native_placement import (
    build_foreign_graft_child,
    randomize_blocks,
    scramble_blocks,
)
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def _recipe():
    return Recipe(
        ParentSpec("synthetic://llama", None, None),
        ParentSpec("synthetic://mamba", None, None),
        LayerSpan(1, 2),
        LayerSpan(1, 2),
        42,
    )


def _parents():
    return tiny_parent("llama", 1), tiny_parent("mamba", 2)


def test_foreign_child_executes_and_freezes_donor():
    recipient, donor = _parents()
    child, manifest = build_foreign_graft_child(
        recipient, donor, recipient_span=(1, 2), donor_span=(1, 2), recipe=_recipe()
    )
    ids = torch.randint(3, 200, (1, 6))
    with torch.no_grad():
        out = child(ids, torch.ones_like(ids))
    assert torch.isfinite(out.logits).all()
    assert manifest["depth_before"] == manifest["depth_after"] == 4
    assert manifest["blocks_removed_unrelated"] == []
    assert [b["action"] for b in manifest["blocks"] if b["source_role"] == "foreign_donor"] == [
        "inserted"
    ]
    trainable = {n for n, p in child.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("input_seam", "output_seam")) for n in trainable)
    child.assert_trainable_contract()


def test_unrelated_recipient_blocks_untouched_and_donor_parent_immutable():
    recipient, donor = _parents()
    before = module_fingerprint(donor)
    child, manifest = build_foreign_graft_child(
        recipient, donor, recipient_span=(1, 2), donor_span=(1, 2), recipe=_recipe()
    )
    rec_blocks = {b["identity"]: b for b in manifest["blocks"] if b["source_role"] == "recipient"}
    assert rec_blocks["recipient:block:1"]["action"] == "replaced"
    assert rec_blocks["recipient:block:0"]["fingerprint_sha256"] == module_fingerprint(
        recipient.model.layers[0]
    )
    assert module_fingerprint(donor) == before


def test_length_changing_graft_rejected():
    recipient, donor = _parents()
    with pytest.raises(ValueError, match="native depth preservation"):
        build_foreign_graft_child(
            recipient, donor, recipient_span=(1, 2), donor_span=(1, 3), recipe=_recipe()
        )


def test_scramble_deterministic_multiset_preserving_and_nonmutating():
    _, donor = _parents()
    blocks = list(donor.backbone.layers)[1:2]
    original = copy.deepcopy(blocks)
    first, record1 = scramble_blocks(blocks, seed=7)
    second, _ = scramble_blocks(blocks, seed=7)
    assert record1["all_multisets_preserved"]
    assert module_fingerprint(first[0]) == module_fingerprint(second[0])
    assert module_fingerprint(blocks[0]) == module_fingerprint(original[0])
    assert not any(p.requires_grad for p in first[0].parameters())


def test_randomize_blocks_deterministic_and_shape_preserving():
    _, donor = _parents()
    blocks = list(donor.backbone.layers)[1:2]
    first, _ = randomize_blocks(blocks, seed=3)
    second, _ = randomize_blocks(blocks, seed=3)
    assert module_fingerprint(first[0]) == module_fingerprint(second[0])
    shapes_before = [tuple(p.shape) for p in blocks[0].parameters()]
    shapes_after = [tuple(p.shape) for p in first[0].parameters()]
    assert shapes_before == shapes_after


def test_bypass_ablation_executes_without_donor():
    recipient, donor = _parents()
    child, _ = build_foreign_graft_child(
        recipient, donor, recipient_span=(1, 2), donor_span=(1, 2), recipe=_recipe()
    )
    ids = torch.randint(3, 200, (1, 5))
    with torch.no_grad():
        out = child(ids, torch.ones_like(ids), ablation="bypass_graft")
    assert torch.isfinite(out.logits).all()
