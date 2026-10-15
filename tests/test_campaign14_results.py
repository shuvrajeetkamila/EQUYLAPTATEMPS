"""Campaign 14 raw-record traceability: gates recomputed from diagnostics."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "campaigns" / "campaign_14"


def diagnostics():
    return json.loads((ROOT / "diagnose_38792029752" / "diagnostics.json").read_text())


def test_baseline_reproduces_campaign13_native_score():
    d = diagnostics()
    v0 = d["variants"]["V0"]
    # Same native recipient macro as Campaign 13's independent jobs.
    assert abs(v0["first"]["macro_dev_nll"] - 2.7731) < 5e-4
    assert v0["macro_repeat_abs_diff"] <= 0.001  # Gate 1
    assert v0["first"]["per_example"] == v0["second"]["per_example"]


def test_exact_native_reconstruction_is_bitwise():
    d = diagnostics()
    for name in ("V1", "V3", "V4"):
        v = d["variants"][name]
        assert v["macro_dev_nll"] == d["variants"]["V0"]["first"]["macro_dev_nll"]
        assert all(x["max_abs_diff"] == 0.0 for x in v["logit_diffs_vs_native"])
        assert v["parameter_audit"]["tensors_changed"] == 0
        assert v["parameter_audit"]["recipient_blocks_removed"] == []


def test_damage_decomposition_ordering():
    d = diagnostics()
    m = {
        k: v.get("macro_dev_nll", v.get("first", {}).get("macro_dev_nll"))
        for k, v in d["variants"].items()
    }
    assert m["V2"] > m["V6"] > m["V5"] > m["V0"]  # seams-alone worst at init
    v5 = d["variants"]["V5"]["parameter_audit"]
    assert v5["recipient_blocks_removed"] == [11, 12]
    assert v5["recipient_blocks_duplicated"] == [24, 25, 26, 27, 28, 29]
    assert d["parents_unchanged"] is True


def test_trained_pair_records_anchor_to_campaign13():
    trained_dirs = sorted(ROOT.glob("trained_*/runs/c14_v8/results.json"))
    assert trained_dirs, "trained records not imported yet"
    record = json.loads(trained_dirs[-1].read_text())
    macro = (
        record["validation"]["groups"]["domain:language"]["token_nll"]
        + record["validation"]["groups"]["domain:math"]["token_nll"]
    ) / 2
    # V8 is the Campaign 13 condition F configuration: must land near 7.4577.
    assert abs(macro - 7.4577) < 0.1
    v7_path = trained_dirs[-1].parent.parent / "c14_v7" / "results.json"
    v7 = json.loads(v7_path.read_text())

    def trained_macro(rec):
        return (
            rec["validation"]["groups"]["domain:language"]["token_nll"]
            + rec["validation"]["groups"]["domain:math"]["token_nll"]
        ) / 2

    def initial_macro(rec):
        return (
            rec["initial_validation"]["groups"]["domain:language"]["token_nll"]
            + rec["initial_validation"]["groups"]["domain:math"]["token_nll"]
        ) / 2

    d = diagnostics()
    # Runner initial evaluations must reproduce the diagnostics V5/V6 scores.
    assert abs(initial_macro(v7) - d["variants"]["V5"]["macro_dev_nll"]) < 0.05
    assert abs(initial_macro(record) - d["variants"]["V6"]["macro_dev_nll"]) < 0.05
    # Descriptive trained comparison (preregistration): at this 32-step
    # budget the historically initialized seams trained further than the
    # identity-initialized ones (V7 8.4623 vs V8 7.4578) despite V7's
    # better start. Recorded as measured; no repair-training claim.
    assert trained_macro(v7) > trained_macro(record)
