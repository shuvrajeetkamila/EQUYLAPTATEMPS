import json

import pytest

from equylaptatemps.cli import main


def read_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_offline_demo_cli_and_report(tmp_path, capsys):
    out, report = tmp_path / "demo", tmp_path / "report.json"
    main(["demo", "--out", str(out), "--steps", "3", "--report", str(report)])
    result = read_stdout(capsys)
    assert result == json.loads(report.read_text())
    assert result["artifact_reload_identical"]
    assert result["frozen_tensors_unchanged"]
    assert not result["pretrained_parent_weights_used"]
    assert not result["real_world_capabilities_validated"]
    assert (out / "model.safetensors").exists()


def test_full_local_weight_workflow(local_parents, tmp_path, capsys):
    recipe, a_path, b_path = local_parents
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()))
    train_data, heldout_data = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    train_data.write_text('{"prompt":"1+1=","answer":"2","domain":"math"}\n')
    heldout_data.write_text('{"prompt":"2+2=","answer":"4","domain":"math"}\n')
    initial, calibrated, searched = (
        tmp_path / name for name in ("initial", "calibrated", "searched")
    )

    main(["validate", str(recipe_path)])
    assert read_stdout(capsys)["local_weights_checked"] is False
    main(["build", "--recipe", str(recipe_path), "--out", str(initial)])
    assert read_stdout(capsys)["training_status"] == "uncalibrated"
    main(
        [
            "train",
            "--child",
            str(initial),
            "--data",
            str(train_data),
            "--validation-data",
            str(heldout_data),
            "--out",
            str(calibrated),
            "--steps",
            "2",
            "--max-length",
            "64",
            "--gradient-checkpointing",
        ]
    )
    assert read_stdout(capsys)["frozen_tensors_unchanged"]
    main(["inspect", str(calibrated), "--verify"])
    assert read_stdout(capsys)["checksums_verified"]
    main(
        [
            "evaluate",
            "--child",
            str(calibrated),
            "--data",
            str(heldout_data),
            "--ablations",
            "--generate-answers",
            "--max-new-tokens",
            "2",
            "--max-length",
            "64",
            "--recipient-baseline",
            str(a_path),
            "--donor-baseline",
            str(b_path),
        ]
    )
    evaluation = read_stdout(capsys)
    assert set(evaluation["ablations"]) == {"bypass_graft", "zero_graft"}
    assert "recipient_baseline" in evaluation and "donor_baseline" in evaluation
    assert 0 <= evaluation["child_answers"]["exact_match"] <= 1
    main(["generate", "--child", str(calibrated), "--prompt", "1+1=", "--max-new-tokens", "2"])
    output = capsys.readouterr()
    assert "not a validated" in output.err
    main(
        [
            "search",
            "--recipe",
            str(recipe_path),
            "--data",
            str(train_data),
            "--validation-data",
            str(heldout_data),
            "--donor-depth",
            "2",
            "--stride",
            "2",
            "--steps",
            "1",
            "--max-length",
            "64",
            "--out",
            str(searched),
        ]
    )
    assert read_stdout(capsys)["candidate_count"] == 2
    assert (searched / "model.safetensors").is_file()


def test_cli_refuses_existing_output_early(tmp_path, capsys):
    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(SystemExit) as error:
        main(["demo", "--out", str(target)])
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err


def test_cli_errors_are_actionable(capsys):
    with pytest.raises(SystemExit) as error:
        main(["validate", "missing.json"])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
