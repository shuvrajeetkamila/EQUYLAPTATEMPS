"""Campaign 12 within-row control + publisher gate mechanics: synthetic, offline."""

import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.controls import row_shuffled_sector_control
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
    child = row_shuffled_sector_control(
        _recipe(), tiny_parent("mamba", 1), donor, shuffle_seed=seed
    )
    after = module_fingerprint(torch.nn.ModuleList(donor.model.layers[1:3]))
    return child, donor, before, after


def test_source_donor_is_never_mutated():
    _, _, before, after = _control()
    assert before == after


def test_every_row_multiset_preserved_shape_dtype_kept():
    child, donor, _, _ = _control()
    record = child.control_record
    assert record["kind"] == "within_row_permutation"
    assert record["all_multisets_preserved"] and record["all_row_multisets_preserved"]
    originals = dict(torch.nn.ModuleList(donor.model.layers[1:3]).named_parameters())
    graft = dict(child.graft.layers.named_parameters())
    assert set(originals) == set(graft)
    for name, param in graft.items():
        source = originals[name]
        assert param.shape == source.shape and param.dtype == source.dtype
        if param.ndim == 2:  # row-wise exact multiset + row statistics preserved
            assert torch.equal(torch.sort(param, dim=1).values, torch.sort(source, dim=1).values)
            assert torch.allclose(param.mean(dim=1), source.mean(dim=1))
            assert torch.allclose(param.norm(dim=1), source.norm(dim=1))
        else:  # excluded 1D tensors preserved bit-exactly
            assert torch.equal(param, source)


def test_arrangement_actually_destroyed_and_fingerprint_reproducible():
    child, donor, _, _ = _control()
    record = child.control_record
    changed = [
        t for t in record["tensors"] if t["eligible"] and t["sha256_before"] != t["sha256_after"]
    ]
    assert len(changed) == record["eligible_tensors"]
    original = torch.nn.ModuleList(donor.model.layers[1:3])
    assert module_fingerprint(original) != module_fingerprint(child.graft.layers)
    child_b, _, _, _ = _control()  # same seed: identical control fingerprint
    assert module_fingerprint(child.graft.layers) == module_fingerprint(child_b.graft.layers)


def test_row_shuffle_deterministic_and_seed_sensitive():
    child_a, _, _, _ = _control()
    child_c, _, _, _ = _control(seed=7)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(child_a.graft.parameters(), child_c.graft.parameters(), strict=True)
    )


def test_row_shuffled_child_is_frozen_and_executes():
    child, _, _, _ = _control()
    assert not any(p.requires_grad for p in child.graft.parameters())
    ids = torch.randint(3, 250, (2, 10))
    labels = ids.clone()
    labels[:, :3] = -100
    assert torch.isfinite(child(ids, torch.ones_like(ids), labels).loss)
    for ablation in ("bypass_graft", "zero_graft"):
        assert torch.isfinite(child(ids, torch.ones_like(ids), labels, ablation=ablation).loss)


def test_benefit_and_gate_mechanics():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "publish_campaign12", Path(__file__).parent.parent / "scripts" / "publish_campaign12.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.benefit_percent(10.0, 8.0) == 20.0  # (control, intact): intact better
    assert module.benefit_percent(8.0, 10.0) == -25.0  # negative: control better
    assert module.benefit_percent(9.0, 9.0) == 0.0  # exact tie
    good = [
        {"seed": 42, "vs_B_percent": 3.0, "vs_C_percent": 2.5},
        {"seed": 123, "vs_B_percent": 1.5, "vs_C_percent": 1.2},
        {"seed": 31415, "vs_B_percent": 1.1, "vs_C_percent": 4.0},
    ]
    assert module.learned_structure_gate(good) is True
    weak123 = [dict(r) for r in good]
    weak123[1]["vs_C_percent"] = 0.5
    assert module.learned_structure_gate(weak123) is False  # one weak seed kills it
    assert module.classify(True, True, True) == "A"
    assert module.classify(True, True, False) == "B"
    assert module.classify(True, False, False) == "C"
    assert module.classify(False, True, True) == "D"
