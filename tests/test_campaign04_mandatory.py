"""Campaign 4 mandatory-donor-use mechanics: donor-delta pathway, bottleneck
interfaces, matched-twin parameter accounting, reference reconstruction.
Tiny synthetic parents only; not capability evidence."""

import torch

from equylaptatemps.parents import LoadedParents
from equylaptatemps.references import load_reference, save_reference
from equylaptatemps.sequence_seams import (
    BottleneckSequenceSeam,
    install_bottleneck_interfaces,
    install_sequence_interfaces,
)

ATTN = {
    "family": "sequence",
    "kind": "causal_attention",
    "bottleneck": 8,
    "heads": 2,
    "max_positions": 32,
}
BN = {
    "family": "bottleneck",
    "kind": "causal_attention",
    "bottleneck": 4,
    "heads": 2,
    "max_positions": 32,
}


def test_bottleneck_seam_is_narrow_and_causal():
    seam = BottleneckSequenceSeam(16, 24, BN)
    assert seam.down.weight.shape == (4, 16) and seam.up.weight.shape == (24, 4)
    with torch.no_grad():
        for parameter in seam.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    first = torch.randn(1, 6, 16)
    second = first.clone()
    second[:, 4:] = torch.randn(1, 2, 16) * 5
    with torch.no_grad():
        torch.testing.assert_close(seam(first)[:, :4], seam(second)[:, :4], rtol=1e-6, atol=1e-6)


def test_donor_delta_removes_identity_donor_signal(child_bundle):
    child, tok, prov, a, b = child_bundle
    install_sequence_interfaces(child, {**ATTN, "donor_delta": True})
    assert child.donor_delta is True
    ids = torch.tensor([[8, 9, 10, 11]])
    captured = {}

    def capture(_module, args):
        captured["suffix_input"] = args[0].detach()
        return None

    handle = child.output_seam.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            full = child(ids).logits
            bypass = child(ids, ablation="bypass_graft").logits
    finally:
        handle.remove()
    # Under bypass the un-subtracted seam signal remains (donor-removed test),
    # while the full path subtracts it; the two computations must differ.
    with torch.inference_mode():
        assert not torch.allclose(full, bypass)


def test_bottleneck_delta_child_reference_roundtrip(child_bundle, tmp_path):
    child, tok, prov, a, b = child_bundle
    design = {**BN, "bottleneck": 8, "donor_delta": True}
    install_bottleneck_interfaces(child, design)
    child.assert_trainable_contract()
    ids = torch.tensor([[8, 9, 10, 11]])
    loss = child(ids, labels=ids.clone()).loss
    loss.backward()
    assert all(
        p.grad is None
        for n, p in child.named_parameters()
        if not n.startswith(("input_seam.", "output_seam."))
    )
    dense_trainable = 2 * (16 * 24 + 24) + 2 * (24 * 16 + 16)
    assert sum(p.numel() for p in child.parameters() if p.requires_grad) != dense_trainable
    save_reference(child, tmp_path / "reference", prov)
    restored = load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))
    assert restored.donor_delta is True
    with torch.inference_mode():
        torch.testing.assert_close(child(ids).logits, restored(ids).logits, rtol=0, atol=0)
