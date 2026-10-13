"""Alignment conditioning mechanics and immutable recovery, not pretrained performance claims."""

import pytest
import torch

from equylaptatemps.alignment import initialize_conditioned, install_interfaces, whitening_matrix
from equylaptatemps.artifacts import load_artifact, save_artifact
from equylaptatemps.campaign_registry import CampaignRegistry
from equylaptatemps.parents import LoadedParents
from equylaptatemps.references import load_reference, save_reference


@pytest.mark.parametrize("architecture", ["layernorm_linear", "linear", "rmsnorm_linear"])
def test_new_interfaces_roundtrip_parent_reference_and_child_only(
    child_bundle, tmp_path, architecture
):
    child, tok, prov, a, b = child_bundle
    install_interfaces(child, {"architecture": architecture, "initialization": "fixed_orthogonal"})
    initialize_conditioned(child, "fixed_orthogonal", seed=42)
    save_reference(child, tmp_path / "reference", prov)
    restored = load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))
    save_artifact(child, tok, tmp_path / "artifact", prov)
    standalone = load_artifact(tmp_path / "artifact").model
    ids = torch.tensor([[8, 9, 10]])
    with torch.inference_mode():
        expected = child(ids).logits
        torch.testing.assert_close(expected, restored(ids).logits, rtol=0, atol=0)
        torch.testing.assert_close(expected, standalone(ids).logits, rtol=0, atol=0)
    child.assert_trainable_contract()


def test_covariance_conditioning_source_moments_without_teacher_vectors():
    cov = torch.diag(torch.tensor([1.0, 4.0, 9.0]))
    w = whitening_matrix(cov, shrinkage=0, floor=0.01)
    torch.testing.assert_close(w @ cov @ w.T, torch.eye(3), rtol=1e-5, atol=1e-5)


def test_true_identity_not_silently_claimed_when_widths_differ(child_bundle):
    child, *_ = child_bundle
    with pytest.raises(ValueError):
        install_interfaces(child, {"architecture": "identity"})


def test_registry_skip_requires_config_and_artifact_verification(tmp_path):
    config = [{"id": "fixture", "seed": 42}]
    r = CampaignRegistry(tmp_path, config)
    artifact = tmp_path / "results.json"
    artifact.write_text("{}")
    r.update("fixture", "COMPLETED", paths=[artifact], metrics={"fixture_not_benchmark": True})
    assert r.verified_complete("fixture")
    artifact.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="tampered"):
        r.verified_complete("fixture")
    with pytest.raises(ValueError, match="Changed campaign registry"):
        CampaignRegistry(tmp_path, [{"id": "fixture", "seed": 43}])


def test_campaign_status_does_not_invent_scientific_outcome(tmp_path):
    r = CampaignRegistry(tmp_path, [{"id": "unrun"}])
    assert r.state["records"]["unrun"]["evaluation_status"] == "NOT_RUN"
    assert r.summary()["outcome"] == "NOT_CLASSIFIED"
