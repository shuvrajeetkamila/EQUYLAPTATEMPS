import pytest

from equylaptatemps.audit import frozen_fingerprint
from equylaptatemps.data import Record
from equylaptatemps.parents import load_parents
from equylaptatemps.search import search_sectors
from equylaptatemps.training import TrainingOptions


def test_sector_search_ranks_complete_blocks_on_disjoint_validation(local_parents):
    recipe, *_ = local_parents
    parents = load_parents(recipe)
    train = [Record("1+1=", "2"), Record("2+2=", "4")]
    validation = [Record("3+3=", "6")]
    progress = []
    best, provenance, report = search_sectors(
        recipe,
        parents,
        train,
        validation,
        TrainingOptions(steps=3, batch_size=2, learning_rate=0.02, max_length=64),
        donor_depth=2,
        stride=2,
        progress=progress.append,
    )
    assert report["candidate_count"] == 2
    assert {tuple(row["donor_sector"]) for row in progress} == {(0, 2), (2, 4)}
    assert report["selected_sector"] == report["candidates"][0]["donor_sector"]
    assert len(best.graft.layers) == 2
    assert report["selected_calibration"]["frozen_tensors_sha256"] == frozen_fingerprint(best)
    assert "source_donor_sector_sha256" in provenance
    assert "NOT the final test" in report["warning"]


def test_search_includes_last_window_when_stride_misses_it(local_parents):
    recipe, *_ = local_parents
    best, _, report = search_sectors(
        recipe,
        load_parents(recipe),
        [Record("a", "b")],
        [Record("c", "d")],
        TrainingOptions(steps=1, max_length=64),
        donor_depth=1,
        stride=2,
    )
    assert {tuple(row["donor_sector"]) for row in report["candidates"]} == {(0, 1), (2, 3), (3, 4)}
    assert len(best.graft.layers) == 1


@pytest.mark.parametrize("depth,stride", [(0, 1), (5, 1), (1, 0), (1, 1.5)])
def test_bad_search_grid_rejected(local_parents, depth, stride):
    recipe, *_ = local_parents
    with pytest.raises(ValueError):
        search_sectors(
            recipe,
            load_parents(recipe),
            [Record("a", "b")],
            [Record("c", "d")],
            TrainingOptions(steps=1, max_length=64),
            donor_depth=depth,
            stride=stride,
        )
