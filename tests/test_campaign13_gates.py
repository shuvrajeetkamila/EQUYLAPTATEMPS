"""Campaign 13 frozen decision rules, tested against hand-computed examples."""

import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "publish_campaign13", Path(__file__).parent.parent / "scripts" / "publish_campaign13.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ablation_criterion_matches_frozen_wording_exactly():
    # Frozen: "bypass and zero ablation macros finite and above intact".
    assert module.ablation_criterion(8.0, 9.0, 10.0) is True
    assert module.ablation_criterion(8.0, 30.0, 20.0) is True  # bypass > zero still passes
    assert module.ablation_criterion(8.0, 7.0, 20.0) is False  # bypass below intact
    assert module.ablation_criterion(8.0, 9.0, 8.0) is False  # zero not above intact
    assert module.ablation_criterion(8.0, math.inf, 20.0) is False
    assert module.ablation_criterion(8.0, math.nan, 20.0) is False


def test_gate_thresholds_per_seed():
    good = [
        {"seed": 42, "vs_B_percent": 2.0, "vs_C_percent": 2.5},
        {"seed": 123, "vs_B_percent": 1.0, "vs_C_percent": 1.2},
        {"seed": 31415, "vs_B_percent": 1.1, "vs_C_percent": 4.0},
    ]
    assert module.learned_structure_gate(good) is True
    weak42 = [dict(r) for r in good]
    weak42[0]["vs_B_percent"] = 1.99
    assert module.learned_structure_gate(weak42) is False  # seed 42 needs >= 2
    weak_new = [dict(r) for r in good]
    weak_new[2]["vs_C_percent"] = 0.99
    assert module.learned_structure_gate(weak_new) is False  # new seed needs >= 1


def test_arrangement_sensitivity_adverse_rule():
    adverse = [
        {"seed": s, "vs_B_percent": -3.0, "vs_C_percent": -2.5} for s in (42, 123, 31415)
    ]
    assert module.arrangement_sensitivity(adverse) == "adverse_detected"
    mixed = [dict(r) for r in adverse]
    mixed[1]["vs_C_percent"] = 0.5
    assert module.arrangement_sensitivity(mixed) == "none"  # one positive breaks it
    shallow = [{"seed": s, "vs_B_percent": -0.5, "vs_C_percent": -0.5} for s in (42, 123, 31415)]
    assert module.arrangement_sensitivity(shallow) == "none"  # pooled mean above -2


def test_classify_mapping():
    assert module.classify(True, True, True, True) == "A"
    assert module.classify(True, True, True, False) == "B"
    assert module.classify(True, False, True, True) == "C"
    assert module.classify(True, True, False, True) == "C"  # no causal support, no pass
    assert module.classify(False, True, True, True) == "D"


def test_benefit_sign_convention():
    assert module.benefit_percent(10.0, 8.0) == pytest.approx(20.0)
    assert module.benefit_percent(8.0, 10.0) == pytest.approx(-25.0)
    assert module.benefit_percent(9.0, 9.0) == 0.0
