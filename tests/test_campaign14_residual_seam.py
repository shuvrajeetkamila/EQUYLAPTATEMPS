"""Campaign 14 residual-identity seam and reconstruction machinery: offline."""

import pytest
import torch

from equylaptatemps.alignment import install_interfaces
from equylaptatemps.model import ResidualDenseSeam, child_from_parents
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def test_residual_seam_is_exact_identity_at_init_and_trainable():
    seam = ResidualDenseSeam(16, 16)
    x = torch.randn(3, 5, 16)
    assert torch.equal(seam(x), x)  # exact identity, not approximate
    loss = (seam(x) ** 2).sum()
    loss.backward()
    # The projection receives nonzero gradients at the identity point.
    assert seam.projection.weight.grad is not None
    assert seam.projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="equal source and destination"):
        ResidualDenseSeam(16, 8)


def _self_recipe(cut, sector):
    return Recipe(
        ParentSpec("synthetic://llama", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(*cut),
        LayerSpan(*sector),
        42,
        positive_control=True,
    )


def test_native_placement_child_with_residual_seams_matches_bypassed():
    recipient, donor = tiny_parent("llama", 1), tiny_parent("llama", 2)
    # Same checkpoint content for both roles: donor copy equals recipient.
    donor.load_state_dict(recipient.state_dict())
    child_a = child_from_parents(_self_recipe((2, 4), (2, 4)), recipient, donor)
    install_interfaces(child_a, {"architecture": "residual_identity"})
    child_b = child_from_parents(_self_recipe((2, 4), (2, 4)), recipient, donor)
    child_b.input_seam = torch.nn.Identity()
    child_b.output_seam = torch.nn.Identity()
    ids = torch.randint(3, 250, (2, 10))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits_a = child_a(ids, mask).logits
        logits_b = child_b(ids, mask).logits
    assert torch.equal(logits_a, logits_b)  # residual seams are exact identity
    child_a.assert_trainable_contract()
    assert child_a.interfaces_config["architecture"] == "residual_identity"


def test_residual_install_rejects_width_changing_child():
    recipe = Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )
    child = child_from_parents(recipe, tiny_parent("mamba", 1), tiny_parent("llama", 2))
    with pytest.raises(ValueError, match="equal source and destination"):
        install_interfaces(child, {"architecture": "residual_identity"})
