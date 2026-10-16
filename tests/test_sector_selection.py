"""Selection labels must follow measured development scores, not names or test performance."""

import copy
import json
from pathlib import Path

import pytest

from equylaptatemps.sectors import annotate_selection

ROOT = Path(__file__).resolve().parents[1] / "reports/pretrained/run-38293921543"


def test_annotate_actual_sector_inventory_without_overwriting_history():
    inventory = json.loads((ROOT / "sector-inventory.json").read_text())["donor"]
    search = json.loads((ROOT / "results.json").read_text())["search"]
    annotated = annotate_selection(inventory, search)
    assert [s["selection_status"] for s in annotated["sectors"]] == ["REJECTED", "SELECTED"]
    assert (
        annotated["sectors"][1]["development_selection_score"]
        == search["candidates"][1]["development_selection_score"]
    )
    assert all(s["selection_status"] == "NOT_RUN" for s in inventory["sectors"])


@pytest.mark.parametrize("failure", ["absent", "wrong_winner", "nonfinite"])
def test_selection_cannot_fabricate_winner(failure):
    inventory = json.loads((ROOT / "sector-inventory.json").read_text())["donor"]
    search = copy.deepcopy(json.loads((ROOT / "results.json").read_text())["search"])
    if failure == "absent":
        search["selected"] = [8, 10]
    elif failure == "wrong_winner":
        search["selected"] = [13, 15]
    else:
        search["candidates"][0]["development_selection_score"] = float("nan")
    with pytest.raises(ValueError):
        annotate_selection(inventory, search)
