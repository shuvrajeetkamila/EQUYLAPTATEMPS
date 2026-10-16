"""Campaign 18 placement-map integrity: offline tests."""

import hashlib
import json
from pathlib import Path

SETS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "campaigns/campaign_18/config/placement_sets.json"
    ).read_text()
)


def test_sets_are_frozen_permutations_and_disjoint():
    ids = []
    for _name, conditions in SETS["sets"].items():
        for cond in conditions:
            ids.append(cond["id"])
            assert sorted(cond["execution_order"]) == list(range(30)), cond["id"]
    assert len(ids) == len(set(ids)) == 83
    blob = json.dumps(SETS["sets"], indent=2, sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == SETS["sets_sha256"]


def test_displacement_bookkeeping():
    for conditions in SETS["sets"].values():
        for cond in conditions:
            order = cond["execution_order"]
            moved = {b: order.index(b) - b for b in range(30) if order.index(b) != b}
            assert cond["n_blocks_displaced"] == len(moved)
            assert cond["max_abs_displacement"] == max((abs(d) for d in moved.values()), default=0)
            assert cond["displacements"] == {str(k): v for k, v in sorted(moved.items())}


def test_move_convention_final_position():
    cond = next(c for c in SETS["sets"]["development"] if c["id"] == "dev-move-b14-d+3")
    assert cond["execution_order"].index(14) == 17
    cond = next(c for c in SETS["sets"]["development"] if c["id"] == "dev-move-b8-d-3")
    assert cond["execution_order"].index(8) == 5


def test_inverse_permutation_restores_identity():
    for conditions in SETS["sets"].values():
        for cond in conditions:
            order = cond["execution_order"]
            inv = [0] * 30
            for position, block in enumerate(order):
                inv[block] = position
            assert [order[inv[i]] for i in range(30)] == list(range(30))


def test_damage_definition_and_predictor_features():
    native = 2.773126
    damage = 11.168408 - native
    assert abs(damage - 8.395282) < 1e-6
    cond = next(c for c in SETS["sets"]["validation"] if c["id"] == "val-xchg-a11-c24-L2")
    assert cond["n_blocks_displaced"] >= 4  # two sections plus shifted middle
    assert cond["max_abs_displacement"] == 13
