"""Campaign 15 frozen decision rules and raw-record traceability."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publish_campaign15 import decide, macro  # noqa: E402

C15 = ROOT / "campaigns" / "campaign_15"
PANELS = {
    42: "panel42_38865626143",
    123: "panel123_38865628701",
    31415: "panel31415_38865631168",
}


def test_decision_rule_precedence_hand_computed():
    win = [1.0, 2.0, 1.5]
    lose = [-1.0, -0.9, -1.1]
    assert decide(False, True, True, win, win) == "IMPLEMENTATION_FAILURE"
    assert decide(True, False, True, win, win) == "INCONCLUSIVE"
    assert decide(True, True, True, win, win) == "REPAIRED_START_ADVANTAGE"
    assert decide(True, True, True, lose, lose) == "INITIALIZATION_ONLY"
    assert decide(True, True, False, lose, lose) == "NO_RELIABLE_ADVANTAGE"
    assert decide(True, True, True, lose, win) == "TRAINING_DEPENDENT"
    assert decide(True, True, True, win, lose) == "TRAINING_DEPENDENT"


def _record(seed, condition):
    path = C15 / PANELS[seed] / "runs" / f"c15_{condition}_{seed}" / "results.json"
    return json.loads(path.read_text())


def test_panel_records_complete_paired_and_frozen():
    for seed in PANELS:
        old, new = _record(seed, "old"), _record(seed, "new")
        for record in (old, new):
            assert record["training"]["execution_status"] == "COMPLETED"
            assert record["training"]["completed_steps"] == 96
            assert record["parent_weights_frozen"] is True
            assert record["structure"]["trainable_dense_seam_parameters"] == 667008
            assert {e["step"] for e in record["checkpoint_evaluations"]} == {0, 8, 16, 32, 64, 96}
        old_order = [tuple(e["record_indices"]) for e in old["training"]["trace"]]
        new_order = [tuple(e["record_indices"]) for e in new["training"]["trace"]]
        assert old_order == new_order  # exact paired data order


def test_published_analysis_matches_raw_records():
    analysis = json.loads((C15 / "paired_analysis.json").read_text())
    assert analysis["outcome"] == "TRAINING_DEPENDENT"
    assert analysis["gate_a_pass"] and analysis["gate_b_pass"]
    assert not analysis["gate_c_pass"]
    for seed in PANELS:
        old, new = _record(seed, "old"), _record(seed, "new")
        old32 = macro(
            next(e["validation"] for e in old["checkpoint_evaluations"] if e["step"] == 32)
        )
        new32 = macro(
            next(e["validation"] for e in new["checkpoint_evaluations"] if e["step"] == 32)
        )
        assert old32 - new32 < 0  # old connector leads at step 32 at every seed
        old0 = macro(old["initial_validation"])
        new0 = macro(new["initial_validation"])
        assert abs(new0 - 12.015005577316103) < 1e-6  # placement-only damage level
        assert old0 - new0 > 0
