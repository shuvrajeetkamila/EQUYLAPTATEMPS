"""Campaign evidence checks use recorded results; they are not new capability measurements."""

import json
import shutil
from pathlib import Path

import pytest

from equylaptatemps.campaign_verify import validate_matrix, validate_replay

ROOT = Path(__file__).resolve().parents[1] / "campaigns/campaign_02"


def test_imported_negative_baseline_records_recompute():
    result = validate_replay(ROOT / "baselines/replay_38354043771")
    assert result["status"] == "PASS"


def test_imported_development_matrix_has_no_test_promotion():
    result = validate_matrix(
        ROOT / "matrix_38358962201",
        external_artifact=ROOT / "matrix_38358962201/external_receipt.json",
    )
    assert result["status"] == "PASS"
    assert result["locally_unverified_binaries"]


def test_matrix_rejects_an_edited_development_metric(tmp_path):
    source = ROOT / "matrix_38358962201"
    target = tmp_path / "matrix"
    shutil.copytree(source, target)
    path = target / "runs/primary_covariance_42/results.json"
    report = json.loads(path.read_text())
    report["validation"]["groups"]["overall"]["token_nll"] = 0.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metric mismatch|Invented validation loss|tampered"):
        validate_matrix(target, external_artifact=target / "external_receipt.json")
