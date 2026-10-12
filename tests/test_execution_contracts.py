"""Offline mutation regressions; tiny parents here are NOT pretrained scientific evidence."""

import json
import shutil
from pathlib import Path

import pytest

from equylaptatemps.contracts import parent_contract
from equylaptatemps.evidence import validate_results
from equylaptatemps.parents import LoadedParents
from equylaptatemps.references import load_reference, save_reference

ROOT = Path(__file__).resolve().parents[1]


def test_reference_rejects_changed_rope_even_when_tensor_hashes_match(child_bundle, tmp_path):
    child, tok, prov, a, b = child_bundle
    save_reference(child, tmp_path / "reference", prov)
    b.config.rope_theta *= 2
    with pytest.raises(ValueError, match="execution contract"):
        load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))


def test_reference_rejects_changed_native_norm_operator(child_bundle, tmp_path):
    child, tok, prov, a, b = child_bundle
    save_reference(child, tmp_path / "reference", prov)
    b.model.layers[1].input_layernorm.variance_epsilon *= 2
    with pytest.raises(ValueError, match="execution contract"):
        load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))


def test_verified_parent_load_contract_rejects_mutable_config(child_bundle, tmp_path):
    child, tok, prov, a, b = child_bundle
    save_reference(child, tmp_path / "reference", prov)
    parents = LoadedParents(
        a,
        b,
        tok,
        prov,
        source_execution_contracts={"recipient": parent_contract(a), "donor": parent_contract(b)},
    )
    b.config.rope_theta *= 2
    with pytest.raises(ValueError, match="changed after verified"):
        load_reference(tmp_path / "reference", parents)


def test_reference_contract_ignores_installation_paths_not_numerical_settings(
    child_bundle, tmp_path
):
    child, tok, prov, a, b = child_bundle
    save_reference(child, tmp_path / "reference", prov)
    a.config._name_or_path = "/a/different/cache/path"
    b.config._name_or_path = "/another/cache/path"
    load_reference(tmp_path / "reference", LoadedParents(a, b, tok, prov))


def test_tampered_inheritance_verdict_rejected_without_changing_old_evidence(tmp_path):
    shutil.copytree(ROOT / "reports/pretrained/run-38293921543", tmp_path / "run")
    path = tmp_path / "run/results.json"
    result = json.loads(path.read_text())
    result["inheritance"]["retention"] = {
        "status": "PASS",
        "kind": "derived",
        "reason": "invented unit-test claim",
    }
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Substituted inheritance"):
        validate_results(tmp_path / "run")


def test_all_comparisons_must_use_same_ordered_examples(tmp_path):
    shutil.copytree(ROOT / "reports/pretrained/run-38293921543", tmp_path / "run")
    root = tmp_path / "run"
    result = json.loads((root / "results.json").read_text())
    entry = result["comparisons"]["calibrated_child"]["gsm8k"]
    file = root / entry["records"]
    rows = file.read_text().splitlines()
    file.write_text("\n".join(reversed(rows)) + "\n")
    from equylaptatemps.audit import sha256_file

    entry["records_sha256"] = sha256_file(file)
    (root / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="different ordered"):
        validate_results(root)
