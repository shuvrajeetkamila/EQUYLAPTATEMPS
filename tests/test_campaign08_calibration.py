"""Campaign 8 calibration: synthetic, offline. Calibration must be a fair,
deterministic, interface-only procedure applied identically to twin children."""

import torch

from equylaptatemps.alignment import initialize_conditioned
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
        "child_input_normalized/covariance": torch.eye(r) + 0.01 * torch.randn(r, r, generator=gen),
        "child_output_normalized/covariance": torch.eye(d)
        + 0.01 * torch.randn(d, d, generator=gen),
    }
    return {"summary": {"streams": streams}, "tensors": tensors}


def test_covariance_calibration_is_deterministic_and_interface_only():
    child_a, child_b = _child(), _child()
    before = {n: p.detach().clone() for n, p in child_a.named_parameters() if "seam" in n}
    stats = _fake_statistics(child_a)
    out_a = initialize_conditioned(child_a, "covariance_orthogonal", statistics=stats, seed=42)
    out_b = initialize_conditioned(child_b, "covariance_orthogonal", statistics=stats, seed=42)
    assert out_a["kind"] == out_b["kind"] == "unpaired_distribution_initialization"
    assert out_a["teacher_targets"] is False
    for (n, p), (m, q) in zip(child_a.named_parameters(), child_b.named_parameters(), strict=True):
        assert n == m and torch.equal(p, q)  # same procedure -> identical seams
    changed = [
        n for n, p in child_a.named_parameters() if "seam" in n and not torch.equal(p, before[n])
    ]
    assert changed  # calibration actually moved the interfaces
    child_a.assert_trainable_contract()  # still seams-only trainable


def test_calibration_requires_statistics():
    child = _child()
    try:
        initialize_conditioned(child, "covariance_orthogonal", statistics=None, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("calibration without measured statistics must be refused")
