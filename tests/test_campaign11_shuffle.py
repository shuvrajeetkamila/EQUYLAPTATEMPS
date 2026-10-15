"""Campaign 11 structure-destroyed control: synthetic, offline."""

import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.controls import shuffled_sector_control
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def _recipe():
    return Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )


def _control(seed=20241014):
    donor = tiny_parent("llama", 2)
    before = module_fingerprint(torch.nn.ModuleList(donor.model.layers[1:3]))
    child = shuffled_sector_control(_recipe(), tiny_parent("mamba", 1), donor, shuffle_seed=seed)
    after = module_fingerprint(torch.nn.ModuleList(donor.model.layers[1:3]))
    return child, donor, before, after


def test_source_donor_is_never_mutated():
    _, _, before, after = _control()
    assert before == after  # deep copy: the pinned parent's sector is untouched


def test_multisets_preserved_norms_excluded_arrangement_destroyed():
    child, donor, _, _ = _control()
    record = child.control_record
    assert record["all_multisets_preserved"]
    assert record["eligible_tensors"] > 0 and record["excluded_tensors"] > 0
    changed = [
        t for t in record["tensors"] if t["eligible"] and t["sha256_before"] != t["sha256_after"]
    ]
    assert len(changed) == record["eligible_tensors"]  # every matrix rearranged
    for t in record["tensors"]:
        if not t["eligible"]:  # RMSNorm gains preserved bit-exactly
            assert t["sha256_before"] == t["sha256_after"]
    # The graft actually differs from the donor's original sector.
    original = torch.nn.ModuleList(donor.model.layers[1:3])
    assert module_fingerprint(original) != module_fingerprint(child.graft.layers)


def test_shuffle_is_deterministic_per_seed():
    child_a, _, _, _ = _control()
    child_b, _, _, _ = _control()
    for p, q in zip(child_a.graft.parameters(), child_b.graft.parameters(), strict=True):
        assert torch.equal(p, q)
    child_c, _, _, _ = _control(seed=7)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(child_a.graft.parameters(), child_c.graft.parameters(), strict=True)
    )


def test_shuffled_child_is_frozen_and_executes():
    child, _, _, _ = _control()
    assert not any(p.requires_grad for p in child.graft.parameters())
    child.assert_trainable_contract()
    ids = torch.randint(3, 250, (2, 10))
    labels = ids.clone()
    labels[:, :3] = -100
    assert torch.isfinite(child(ids, torch.ones_like(ids), labels).loss)
    for ablation in ("bypass_graft", "zero_graft"):
        assert torch.isfinite(child(ids, torch.ones_like(ids), labels, ablation=ablation).loss)
