from pathlib import Path
from zipfile import ZipFile

from equylaptatemps.artifacts import load_artifact
from equylaptatemps.audit import sha256_file


def test_exact_distributed_synthetic_archive_is_valid_and_self_contained(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    archive = examples / "equylaptatemps-synthetic-child-v0.1.0.zip"
    expected, name = archive.with_suffix(".zip.sha256").read_text().split()
    assert name == archive.name
    assert sha256_file(archive) == expected
    with ZipFile(archive) as zipped:
        assert {"child/child.json", "child/model.safetensors", "RUN.md", "LICENSE"} <= set(
            zipped.namelist()
        )
        zipped.extractall(tmp_path)
    stored = load_artifact(tmp_path / "child")
    assert stored.manifest["capability_status"] == "synthetic_smoke_test"
    assert stored.model.recipe.synthetic
    assert stored.model.recipe.recipient.local_path is None
    assert stored.model.recipe.donor.local_path is None
    assert stored.manifest["training_status"] == "seam_calibrated"
    assert stored.model.structure_report()["trainable_dense_seam_parameters"] == 888
