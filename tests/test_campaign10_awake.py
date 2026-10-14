"""Campaign 10: trained-output interface + awake twin calibration (synthetic)."""

import torch

from equylaptatemps.alignment import initialize_conditioned
from equylaptatemps.controls import (
    calibrate_awake_sector,
    graft_activity,
    native_sector_activity,
    random_sector_control,
)
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent
from equylaptatemps.tokenization import ByteTokenizer


def _recipe():
    return Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )


def _child():
    return child_from_parents(_recipe(), tiny_parent("mamba", 1), tiny_parent("llama", 2))


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


def test_trained_output_init_geometrizes_input_only():
    child_a, child_b = _child(), _child()
    stats = _fake_statistics(child_a)
    default_out = {k: v.clone() for k, v in child_a.output_seam.state_dict().items()}
    out = initialize_conditioned(child_a, "natgeom_in_trained_out", statistics=stats, seed=42)
    initialize_conditioned(child_b, "natgeom_in_trained_out", statistics=stats, seed=42)
    assert out["output_pathway"] == "trained_default_init_no_geometry"
    # Output seam is exactly the generic constructed default (trained pathway).
    for k, v in child_a.output_seam.state_dict().items():
        assert torch.equal(v, default_out[k])
    # Input seam did receive donor-native geometry (deterministic across children).
    assert torch.equal(child_a.input_seam.projection.weight, child_b.input_seam.projection.weight)
    child_a.assert_trainable_contract()


def test_graft_activity_and_native_sector_activity_measure():
    donor = tiny_parent("llama", 5)
    tok = ByteTokenizer()
    native = native_sector_activity(donor, tok, ["hello world", "surgery test"], LayerSpan(1, 3))
    assert native["texts"] == 2 and native["mean_relative_update"] > 0
    twin = random_sector_control(_recipe(), tiny_parent("mamba", 1), donor, initialization_seed=7)
    act = graft_activity(twin, tok, ["hello world"])
    assert act["all_finite"] and act["mean_relative_update"] is not None


def test_awake_calibration_moves_activity_toward_target_and_stays_frozen():
    donor = tiny_parent("llama", 5)
    tok = ByteTokenizer()
    twin = random_sector_control(_recipe(), tiny_parent("mamba", 1), donor, initialization_seed=7)
    texts = ["hello world", "calibration sample"]
    before = graft_activity(twin, tok, texts)["mean_relative_update"]
    target = max(before * 3.0, 0.05)
    record = calibrate_awake_sector(twin, tok, texts, target=target)
    after = graft_activity(twin, tok, texts)["mean_relative_update"]
    assert record["target_relative_update"] == target
    assert after > before  # scaling residual branches increased sector activity
    assert not any(p.requires_grad for p in twin.graft.parameters())
    twin.assert_trainable_contract()


def test_awake_calibration_records_unmeasurable_target():
    donor = tiny_parent("llama", 5)
    tok = ByteTokenizer()
    twin = random_sector_control(_recipe(), tiny_parent("mamba", 1), donor, initialization_seed=7)
    record = calibrate_awake_sector(twin, tok, ["x"], target=float("nan"))
    assert record["calibrated"] is False
