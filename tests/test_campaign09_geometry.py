"""Campaign 9 donor-native geometry interface: synthetic, offline."""

import torch

from equylaptatemps.alignment import coloring_matrix, initialize_conditioned
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def _child(seed=42):
    recipe = Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        seed,
    )
    return child_from_parents(recipe, tiny_parent("mamba", 1), tiny_parent("llama", 2))


def _fake_statistics(child):
    r = child.recipient_config.hidden_size
    d = child.donor_config.hidden_size
    gen = torch.Generator().manual_seed(0)

    def cov(n, scale=1.0):
        a = torch.randn(n, n, generator=gen)
        return scale * (a @ a.T / n + torch.eye(n))

    streams = {
        name: {"global_rms": 2.0}
        for name in (
            "donor_native_input",
            "recipient_native_output",
            "child_donor_input",
            "child_host_suffix_input",
        )
    }
    tensors = {
        "child_input_normalized/covariance": cov(r),
        "child_output_normalized/covariance": cov(d),
        "donor_native_input/covariance": cov(d, scale=50.0),
        "recipient_native_output/covariance": cov(r, scale=2.0),
    }
    return {"summary": {"streams": streams}, "tensors": tensors}


def test_coloring_matrix_is_symmetric_square_root():
    gen = torch.Generator().manual_seed(1)
    a = torch.randn(8, 8, generator=gen)
    cov = a @ a.T + torch.eye(8)
    root = coloring_matrix(cov)
    assert torch.allclose(root, root.T, atol=1e-5)
    assert torch.allclose(root @ root, cov, atol=1e-3)


def test_geometry_init_deterministic_interface_only_and_recorded():
    child_a, child_b = _child(), _child()
    stats = _fake_statistics(child_a)
    out_a = initialize_conditioned(child_a, "donor_native_geometry", statistics=stats, seed=42)
    out_b = initialize_conditioned(child_b, "donor_native_geometry", statistics=stats, seed=42)
    assert out_a["method"] == out_b["method"] == "donor_native_geometry"
    assert out_a["teacher_targets"] is False
    for (n, p), (m, q) in zip(child_a.named_parameters(), child_b.named_parameters(), strict=True):
        assert n == m and torch.equal(p, q)
    child_a.assert_trainable_contract()


def test_geometry_init_actually_transforms_the_seams():
    child = _child()
    before = {n: p.detach().clone() for n, p in child.named_parameters() if "seam" in n}
    initialize_conditioned(
        child, "donor_native_geometry", statistics=_fake_statistics(child), seed=42
    )
    changed = [
        n for n, p in child.named_parameters() if "seam" in n and not torch.equal(p, before[n])
    ]
    assert changed


def test_geometry_init_requires_statistics():
    child = _child()
    try:
        initialize_conditioned(child, "donor_native_geometry", statistics=None, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("geometry init without measured statistics must be refused")
