"""Native-shape unit checks; random sectors are NOT inheritance evidence."""

import pytest
import torch

from equylaptatemps.artifacts import save_artifact
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.controls import random_qwen_sector_control
from equylaptatemps.objectives import encode_text
from equylaptatemps.references import save_reference
from equylaptatemps.training import TrainingOptions, fit_encoded_seams


def test_random_sector_preserves_shape_and_original_parent_values(child_bundle):
    child, _, _, a, b = child_bundle
    before = module_fingerprint(b)
    control = random_qwen_sector_control(child.recipe, a, b, initialization_seed=1042)
    assert control.structure_report() == child.structure_report()
    assert module_fingerprint(control.graft.layers) != module_fingerprint(child.graft.layers)
    assert module_fingerprint(b) == before
    assert all(not p.requires_grad for p in control.graft.parameters())
    torch.testing.assert_close(
        control.input_seam.projection.weight, child.input_seam.projection.weight
    )
    with torch.inference_mode():
        assert torch.isfinite(control(torch.tensor([[8, 9, 10]])).logits).all()


def test_random_sector_cannot_be_exported_as_pretrained(child_bundle, tmp_path):
    child, tok, prov, a, b = child_bundle
    control = random_qwen_sector_control(child.recipe, a, b, initialization_seed=1042)
    with pytest.raises(ValueError, match="controls cannot be exported"):
        save_reference(control, tmp_path / "reference", prov)
    with pytest.raises(ValueError, match="controls cannot be exported"):
        save_artifact(control, tok, tmp_path / "child", prov)


def test_matched_control_only_trains_interfaces_and_is_labeled_control(child_bundle):
    child, tok, _, a, b = child_bundle
    control = random_qwen_sector_control(child.recipe, a, b, initialization_seed=1042)
    before = module_fingerprint(control.graft.layers)
    report = fit_encoded_seams(
        control,
        tok,
        encode_text(["Native control unit fixture."], tok, 64),
        TrainingOptions(steps=2, max_length=64, batch_size=1),
    )
    assert report["parent_kind"] == "random_sector_control"
    assert before == module_fingerprint(control.graft.layers)
    assert len(report["trainable_tensors"]) == 8
