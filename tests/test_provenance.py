import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from equylaptatemps.artifacts import load_artifact, save_artifact
from equylaptatemps.audit import sha256_file
from equylaptatemps.parents import build_local, load_parents
from equylaptatemps.provenance import SOURCE_RECORD, snapshot_evidence, snapshot_files


def record_snapshot(spec, path):
    inventory = {name: sha256_file(file) for name, file in snapshot_files(path).items()}
    record = {
        "format_version": 1,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "files": inventory,
        "hub_lfs_sha256": {},
    }
    (path / SOURCE_RECORD).write_text(json.dumps(record))
    return record


def test_locally_recorded_snapshot_inventory_is_verified(local_parents):
    recipe, a_path, _ = local_parents
    record_snapshot(recipe.recipient, a_path)
    evidence = snapshot_evidence(recipe.recipient, a_path)
    assert evidence["revision_evidence"] == "locally_verified_pinned_download_record"
    assert evidence["signed_attestation"] is False
    assert "config.json" in evidence["snapshot_files_sha256"]
    (a_path / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_parents(recipe)


def test_missing_or_added_snapshot_files_are_rejected(local_parents):
    recipe, a_path, _ = local_parents
    record_snapshot(recipe.recipient, a_path)
    (a_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        snapshot_evidence(recipe.recipient, a_path)


def test_source_identity_revision_and_remote_hashes_checked(local_parents):
    recipe, a_path, _ = local_parents
    record = record_snapshot(recipe.recipient, a_path)
    record["revision"] = "incorrect"
    source = a_path / SOURCE_RECORD
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="revision"):
        snapshot_evidence(recipe.recipient, a_path)
    record["revision"] = None
    record["hub_lfs_sha256"] = {"model.safetensors": "incorrect"}
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Hub LFS"):
        snapshot_evidence(recipe.recipient, a_path)


def test_download_cache_excluded_from_inventory(local_parents):
    _, a_path, _ = local_parents
    cache = a_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download.tmp").write_text("temporary")
    assert not any(".cache" in name for name in snapshot_files(a_path))


def test_parent_notices_copied_and_verified_without_parent_paths(local_parents, tmp_path):
    recipe, a_path, b_path = local_parents
    (a_path / "LICENSE").write_text("Synthetic test fixture license A.\n")
    (b_path / "NOTICE.txt").write_text("Synthetic test fixture notice B.\n")
    child, tokenizer, provenance = build_local(recipe)
    target = tmp_path / "child-with-notices"
    save_artifact(child, tokenizer, target, provenance)
    assert (target / "parent-notices/recipient/LICENSE").read_text() == (
        a_path / "LICENSE"
    ).read_text()
    assert load_artifact(target).model.structure_report() == child.structure_report()
    (target / "parent-notices/donor/NOTICE.txt").write_text("tampered")
    with pytest.raises(ValueError, match="notice integrity"):
        load_artifact(target)


def test_notice_filename_cannot_escape_artifact(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    provenance["snapshots"] = {
        "recipient": {"license_notices": {"../../outside.txt": {"text": "bad", "sha256": "bad"}}}
    }
    with pytest.raises(ValueError, match="Unsafe"):
        save_artifact(child, tokenizer, tmp_path / "child", provenance)
    assert not (tmp_path / "outside.txt").exists()


def load_acquisition_script():
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_parents.py"
    spec = importlib.util.spec_from_file_location("prepare_parents_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acquisition_records_real_returned_revision_and_lfs_hash_without_network(
    tmp_path, monkeypatch
):
    module = load_acquisition_script()
    commit = "a1" * 20  # mocked API result, not a claimed real parent commit
    weight_bytes = b"mock safetensors file; never loaded as a model"
    config_bytes = b"{}"
    import hashlib

    expected_hash = hashlib.sha256(weight_bytes).hexdigest()
    siblings = [
        SimpleNamespace(rfilename="model.safetensors", lfs=SimpleNamespace(sha256=expected_hash)),
        SimpleNamespace(rfilename="config.json", lfs=None),
    ]
    info = SimpleNamespace(sha=commit, siblings=siblings)
    monkeypatch.setattr(module, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **kw: info))
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(weight_bytes)
        (path / "config.json").write_bytes(config_bytes)

    monkeypatch.setattr(module, "snapshot_download", fake_download)
    result = module.acquire("state-spaces/mamba-130m-hf", tmp_path)
    assert result.revision == calls[0]["revision"] == commit
    assert not any("bin" in pattern for pattern in calls[0]["allow_patterns"])
    record = json.loads((Path(result.local_path) / SOURCE_RECORD).read_text())
    assert record["hub_lfs_sha256"]["model.safetensors"] == expected_hash
    assert record["files"]["model.safetensors"] == expected_hash


def test_acquisition_refuses_pickle_only_models(tmp_path, monkeypatch):
    module = load_acquisition_script()
    info = SimpleNamespace(
        sha="a1" * 20, siblings=[SimpleNamespace(rfilename="pytorch_model.bin", lfs=None)]
    )
    monkeypatch.setattr(module, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **kw: info))
    with pytest.raises(ValueError, match="refuses pickle"):
        module.acquire("state-spaces/mamba-130m-hf", tmp_path)


def test_parent_notice_bytes_preserved_for_crlf_documents(local_parents, tmp_path):
    recipe, a_path, _ = local_parents
    content = b"Fixture copyright.\r\nFixture license.\r\n"
    (a_path / "LICENSE").write_bytes(content)
    child, tokenizer, provenance = build_local(recipe)
    save_artifact(child, tokenizer, tmp_path / "with-crlf", provenance)
    assert (tmp_path / "with-crlf/parent-notices/recipient/LICENSE").read_bytes() == content
    assert load_artifact(tmp_path / "with-crlf").manifest["provenance"] == provenance
