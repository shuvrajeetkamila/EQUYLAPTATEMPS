"""Campaign 15 instrumentation: segment-driven training must match uninterrupted."""

from dataclasses import dataclass

import torch

from equylaptatemps.alignment import install_interfaces
from equylaptatemps.campaign_training import fit_resumable
from equylaptatemps.data import EncodedRecord
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent
from equylaptatemps.training import TrainingOptions


@dataclass
class FakeTokenizer:
    pad_token_id: int = 0


def _child():
    recipient, donor = tiny_parent("llama", 1), tiny_parent("llama", 2)
    donor.load_state_dict(recipient.state_dict())
    recipe = Recipe(
        ParentSpec("synthetic://llama", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 2),
        LayerSpan(2, 3),
        42,
        positive_control=True,
    )
    child = child_from_parents(recipe, recipient, donor)
    install_interfaces(child, {"architecture": "residual_identity"})
    return child


def _records():
    return [EncodedRecord([1, 5, 6, 7, 2], [-100, 5, 6, 7, 2], "language") for _ in range(6)]


def _fit(child, out, interrupt_after=None):
    return fit_resumable(
        child,
        FakeTokenizer(),
        _records(),
        TrainingOptions(steps=4, seed=7, learning_rate=1e-3),
        validation=_records(),
        output=out,
        contract={"test": "campaign15_segmentation"},
        interrupt_after=interrupt_after,
    )


def test_segmented_fit_matches_uninterrupted(tmp_path):
    full = _child()
    report_full = _fit(full, tmp_path / "full")
    segmented = _child()
    _fit(segmented, tmp_path / "seg", interrupt_after=2)
    report_seg = _fit(segmented, tmp_path / "seg")
    assert report_full["completed_steps"] == report_seg["completed_steps"] == 4
    assert report_full["execution_status"] == report_seg["execution_status"] == "COMPLETED"
    for (n1, p1), (n2, p2) in zip(
        full.named_parameters(), segmented.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-7), n1
    # The interrupted segment itself must report as incomplete, not completed.
    partial = _child()
    report_partial = _fit(partial, tmp_path / "partial", interrupt_after=2)
    assert report_partial["execution_status"] == "INTERRUPTED"
    assert report_partial["completed_steps"] == 2
