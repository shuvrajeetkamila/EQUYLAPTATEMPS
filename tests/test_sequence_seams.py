"""Campaign 3 sequence-aware interface mechanics: causality, identity-at-init,
trainable contract, gradient flow, and bridge-only reference reconstruction.
These are engineering tests on tiny synthetic parents, not capability evidence."""

import pytest
import torch

from equylaptatemps.model import DenseSeam
from equylaptatemps.parents import LoadedParents
from equylaptatemps.references import load_reference, save_reference
from equylaptatemps.sequence_seams import (
    SEQUENCE_KINDS,
    SequenceSeam,
    install_sequence_interfaces,
    sequence_design_report,
)

DESIGNS = [
    {"family": "sequence", "kind": "causal_conv", "kernel_size": 3},
    {
        "family": "sequence",
        "kind": "causal_attention",
        "bottleneck": 8,
        "heads": 2,
        "max_positions": 32,
    },
    {"family": "sequence", "kind": "bottleneck_gru", "bottleneck": 8},
]


@pytest.mark.parametrize("design", DESIGNS)
def test_sequence_seam_equals_dense_seam_at_initialization(design):
    torch.manual_seed(7)
    dense = DenseSeam(16, 24)
    torch.manual_seed(7)
    sequence = SequenceSeam(16, 24, design)
    hidden = torch.randn(2, 6, 16)
    with torch.no_grad():
        torch.testing.assert_close(sequence(hidden), dense(hidden), rtol=0, atol=0)
        mixer_delta = sequence.mixer(sequence.projection(sequence.norm(hidden)))
    assert float(mixer_delta.abs().max()) == 0.0


@pytest.mark.parametrize("kind", SEQUENCE_KINDS)
def test_sequence_mixers_are_causal_after_randomization(kind):
    design = {"family": "sequence", "kind": kind, "kernel_size": 3, "bottleneck": 8, "heads": 2}
    torch.manual_seed(3)
    seam = SequenceSeam(16, 24, design)
    with torch.no_grad():
        for parameter in seam.mixer.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    first = torch.randn(1, 6, 16)
    second = first.clone()
    second[:, 4:] = torch.randn(1, 2, 16) * 5
    with torch.no_grad():
        torch.testing.assert_close(seam(first)[:, :4], seam(second)[:, :4], rtol=1e-6, atol=1e-6)
        assert not torch.allclose(seam(first)[:, 4:], seam(second)[:, 4:])


@pytest.mark.parametrize("design", DESIGNS)
def test_sequence_child_contract_gradients_and_reference_roundtrip(child_bundle, tmp_path, design):
    child, tok, prov, a, b = child_bundle
    install_sequence_interfaces(child, design)
    child.assert_trainable_contract()
    report = sequence_design_report(child)
    assert report["parent_blocks_modified"] is False
    assert report["sequence_mixer_parameters"] > 0
    assert report["trainable_parameters"] == sum(
        p.numel() for p in child.parameters() if p.requires_grad
    )
    ids = torch.tensor([[8, 9, 10, 11]])
    loss = child(ids, labels=ids.clone()).loss
    loss.backward()
    for seam in (child.input_seam, child.output_seam):
        grads = [p.grad for n, p in seam.named_parameters() if n.startswith("mixer.")]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(
        p.grad is None
        for n, p in child.named_parameters()
        if not n.startswith(("input_seam.", "output_seam."))
    )
    save_reference(child, tmp_path / "reference", prov)
    restored = load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))
    with torch.inference_mode():
        torch.testing.assert_close(child(ids).logits, restored(ids).logits, rtol=0, atol=0)


def test_unregistered_sequence_design_is_rejected(child_bundle):
    child, *_ = child_bundle
    with pytest.raises(ValueError):
        install_sequence_interfaces(child, {"family": "sequence", "kind": "future_leaking"})
    with pytest.raises(ValueError):
        install_sequence_interfaces(child, {"kind": "causal_conv"})
