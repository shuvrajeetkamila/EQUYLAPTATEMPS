"""Mechanical fixture tests only; no synthetic capability scores claimed by Campaign2."""


import pytest
import torch

from equylaptatemps.alignment_stats import StreamMoments
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.campaign_training import fit_resumable
from equylaptatemps.objectives import encode_text
from equylaptatemps.synthetic import demo_recipe, make_synthetic_child
from equylaptatemps.training import TrainingOptions


def test_aggregate_statistics_covariance_and_positions():
    meter = StreamMoments(3)
    x = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])
    meter.update(x)
    summary = meter.summary()
    assert summary["vectors"] == 2 and summary["nonfinite"] == 0
    assert summary["channel_mean"] == [1.5, 3.0, 4.5]
    torch.testing.assert_close(
        meter.covariance_matrix(),
        torch.tensor([[0.25, 0.5, 0.75], [0.5, 1.0, 1.5], [0.75, 1.5, 2.25]], dtype=torch.float64),
    )
    assert not hasattr(meter, "activation_targets")


def test_nonfinite_representation_aborts():
    with pytest.raises(FloatingPointError):
        StreamMoments(2).update(torch.tensor([[[float("nan"), 0.0]]]))


def make_fixture():
    child, tok, _, _ = make_synthetic_child(demo_recipe())
    return child, tok


def test_resume_matches_uninterrupted_new_parameter_updates(tmp_path):
    a, tok = make_fixture()
    b, _ = make_fixture()
    records = encode_text(["A short fixture.", "Another short record."], tok, 64)
    opts = TrainingOptions(steps=4, max_length=64, batch_size=1, seed=42)
    fit_resumable(
        a,
        tok,
        records,
        opts,
        validation=records,
        output=tmp_path / "full",
        contract={"fixture": True},
    )
    first = fit_resumable(
        b,
        tok,
        records,
        opts,
        validation=records,
        output=tmp_path / "resume",
        contract={"fixture": True},
        interrupt_after=2,
    )
    assert first["execution_status"] == "INTERRUPTED"
    second = fit_resumable(
        b,
        tok,
        records,
        opts,
        validation=records,
        output=tmp_path / "resume",
        contract={"fixture": True},
    )
    assert second["execution_status"] == "COMPLETED"
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    assert module_fingerprint(a.graft) == module_fingerprint(b.graft)


def test_resume_rejects_changed_data_or_configuration(tmp_path):
    child, tok = make_fixture()
    rows = encode_text(["An offline record."], tok, 64)
    opts = TrainingOptions(steps=2, max_length=64, batch_size=1)
    fit_resumable(
        child,
        tok,
        rows,
        opts,
        validation=rows,
        output=tmp_path / "state",
        contract={"case": "original"},
        interrupt_after=1,
    )
    with pytest.raises(ValueError, match="Unsafe training resume"):
        fit_resumable(
            child,
            tok,
            rows,
            opts,
            validation=rows,
            output=tmp_path / "state",
            contract={"case": "changed"},
        )
