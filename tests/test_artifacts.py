import json

import pytest
import torch

from equylaptatemps.artifacts import load_artifact, save_artifact
from equylaptatemps.audit import module_fingerprint
from equylaptatemps.model import child_from_parents
from equylaptatemps.parents import build_local
from equylaptatemps.synthetic import demo_recipe, tiny_parent
from equylaptatemps.tokenization import ByteTokenizer


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_portable_artifact_exact_roundtrip(tmp_path, reverse, dtype):
    recipe = demo_recipe(reverse=reverse)
    a = tiny_parent(recipe.recipient.identity.family, 11).to(dtype)
    b = tiny_parent(recipe.donor.identity.family, 22).to(dtype)
    child = child_from_parents(recipe, a, b)
    tokenizer = ByteTokenizer()
    provenance = recipe.stranger_report()
    provenance["source_donor_sector_sha256"] = module_fingerprint(child.graft.layers)
    target = tmp_path / "child"
    manifest = save_artifact(child, tokenizer, target, provenance)
    assert manifest["capability_status"] == "synthetic_smoke_test"
    assert manifest["training_status"] == "uncalibrated"
    # Parents may disappear; standalone load reconstructs ONLY the retained modules.
    del a, b
    restored = load_artifact(target)
    ids = torch.tensor([[8, 9, 10, 11]])
    with torch.inference_mode():
        assert torch.equal(child(ids).logits, restored.model(ids).logits)
    if recipe.recipient.identity.family == "mamba":
        assert restored.model.lm_head.weight is restored.model.embedding.weight
    assert restored.manifest["audit"] == manifest["audit"]
    assert restored.tokenizer.encode("hello") == tokenizer.encode("hello")


def test_local_parent_loading_and_exact_sector_verification(local_parents, tmp_path):
    recipe, _, _ = local_parents
    child, tokenizer, provenance = build_local(recipe)
    assert provenance["source_donor_sector_sha256"] == module_fingerprint(child.graft.layers)
    save_artifact(child, tokenizer, tmp_path / "built", provenance)
    assert load_artifact(tmp_path / "built").model.structure_report() == child.structure_report()


def test_changed_weight_bytes_are_rejected(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    save_artifact(child, tokenizer, tmp_path / "child", provenance)
    weights = tmp_path / "child/model.safetensors"
    with weights.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)[0]
        handle.seek(-1, 2)
        handle.write(bytes([value ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(tmp_path / "child")


def test_changed_tokenizer_is_rejected(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    save_artifact(child, tokenizer, tmp_path / "child", provenance)
    ByteTokenizer(4).save_pretrained(str(tmp_path / "child/tokenizer"))
    with pytest.raises(ValueError, match="tokenizer contract"):
        load_artifact(tmp_path / "child")


def test_changed_audit_is_rejected(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    save_artifact(child, tokenizer, tmp_path / "child", provenance)
    path = tmp_path / "child/child.json"
    manifest = json.loads(path.read_text())
    manifest["audit"]["donor_sector_sha256"] = "edited"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit mismatch"):
        load_artifact(tmp_path / "child")


def test_existing_output_never_overwritten(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    target = tmp_path / "existing"
    target.mkdir()
    (target / "sentinel.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        save_artifact(child, tokenizer, target, provenance)
    assert (target / "sentinel.txt").read_text() == "keep"
    assert list(target.iterdir()) == [target / "sentinel.txt"]


def test_atomic_save_cleans_up_on_failure(child_bundle, tmp_path, monkeypatch):
    child, tokenizer, provenance, *_ = child_bundle

    def fail(path):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(tokenizer, "save_pretrained", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_artifact(child, tokenizer, tmp_path / "child", provenance)
    assert not list(tmp_path.iterdir())


def test_changed_donor_cannot_be_exported(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    with torch.no_grad():
        next(child.graft.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="Donor weights changed"):
        save_artifact(child, tokenizer, tmp_path / "child", provenance)


def test_trainable_parent_breaks_contract(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    next(child.graft.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="contract"):
        save_artifact(child, tokenizer, tmp_path / "child", provenance)


def test_modified_rotary_config_cannot_silently_change_execution(child_bundle, tmp_path):
    child, tokenizer, provenance, *_ = child_bundle
    save_artifact(child, tokenizer, tmp_path / "child", provenance)
    path = tmp_path / "child/child.json"
    manifest = json.loads(path.read_text())
    manifest["donor_config"]["rope_theta"] *= 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit mismatch"):
        load_artifact(tmp_path / "child")


def test_malformed_manifest_reports_a_validation_error(tmp_path):
    from equylaptatemps.artifacts import read_manifest

    (tmp_path / "child.json").write_text('{"format_version": 1}')
    with pytest.raises(ValueError, match="missing required"):
        read_manifest(tmp_path)
