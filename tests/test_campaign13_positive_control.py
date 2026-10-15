"""Campaign 13 same-family positive-control opt-in: synthetic, offline."""

import pytest
import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def _recipe(positive_control):
    return Recipe(
        ParentSpec("synthetic://llama", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
        positive_control=positive_control,
    )


def test_cousin_rejection_is_still_the_default():
    with pytest.raises(ValueError, match="Cousin"):
        _recipe(False).validate()


def test_positive_control_opt_in_permits_same_family_and_roundtrips():
    recipe = _recipe(True)
    recipe.validate()  # must not raise
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    report = recipe.stranger_report()
    assert report["positive_control"] is True
    assert report["different_architectures"] is False
    assert report["mode"] == "same_family_positive_control"


def test_stranger_report_stays_honest_by_default():
    recipe = Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )
    report = recipe.stranger_report()
    assert report["different_architectures"] is True
    assert report["positive_control"] is False
    assert report["mode"] == "cross_family_stranger"
    assert Recipe.from_dict(recipe.to_dict()) == recipe


def test_same_family_child_constructs_and_executes():
    recipe = _recipe(True)
    recipient, donor = tiny_parent("llama", 1), tiny_parent("llama", 2)
    before = module_fingerprint(donor)
    child = child_from_parents(recipe, recipient, donor)
    assert module_fingerprint(donor) == before
    assert child.structure_report()["positive_control"] is True
    assert child.structure_report()["recipient_family"] == "llama"
    assert child.structure_report()["donor_family"] == "llama"
    ids = torch.randint(3, 250, (2, 10))
    labels = ids.clone()
    labels[:, :3] = -100
    assert torch.isfinite(child(ids, torch.ones_like(ids), labels).loss)
    for ablation in ("bypass_graft", "zero_graft"):
        assert torch.isfinite(child(ids, torch.ones_like(ids), labels, ablation=ablation).loss)
