import json
from dataclasses import replace

import pytest

from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import demo_recipe, tiny_parent


@pytest.mark.parametrize("reverse", [False, True])
def test_strangers_are_different_families(reverse):
    recipe = demo_recipe(reverse=reverse)
    recipe.validate()
    assert recipe.recipient.identity.family != recipe.donor.identity.family
    assert not recipe.recipient.identity.lineage & recipe.donor.identity.lineage
    assert recipe.stranger_report()["synthetic"] is True
    assert Recipe.from_dict(recipe.to_dict()) == recipe


def test_cousin_qwen_models_rejected():
    # Deliberately fictitious revisions exercise syntax only; no pretrained provenance claimed.
    recipient = ParentSpec("Qwen/Qwen2.5-Math-1.5B", "not-a-snapshot", "a1" * 20)
    donor = ParentSpec("Qwen/Qwen2.5-Math-7B", "not-a-snapshot", "b2" * 20)
    with pytest.raises(ValueError, match="Cousin"):
        Recipe(recipient, donor, LayerSpan(0, 2), LayerSpan(0, 2)).validate()


def test_pretrained_stranger_plan_checks_only_declarations():
    a = ParentSpec("state-spaces/mamba-130m-hf", "not-a-snapshot", "a1" * 20)
    b = ParentSpec("Qwen/Qwen2.5-Math-1.5B", "not-a-snapshot", "b2" * 20)
    recipe = Recipe(a, b, LayerSpan(0, 2), LayerSpan(0, 2))
    report = recipe.stranger_report()
    assert report["synthetic"] is False
    assert "not remotely attested" in report["limitation"]


@pytest.mark.parametrize("revision", [None, "main", "PLACEHOLDER", "0" * 40, "A1" * 20])
def test_unpinned_or_placeholder_revisions_rejected(revision):
    with pytest.raises(ValueError, match="revision|placeholder"):
        ParentSpec("state-spaces/mamba-130m-hf", "models/local", revision).validate()


def test_unknown_model_has_no_provenance_bypass():
    with pytest.raises(ValueError, match="Unsupported parent"):
        ParentSpec("some-user/unreviewed-qwen-finetune", "models/local", "a1" * 20).validate()


def test_synthetic_pretrained_mixture_cannot_claim_pretrained_child():
    recipe = replace(
        demo_recipe(), donor=ParentSpec("Qwen/Qwen2.5-Math-1.5B", "models/local", "a1" * 20)
    )
    with pytest.raises(ValueError, match="mixture"):
        recipe.validate()


@pytest.mark.parametrize("bounds", [[0, 0], [-1, 2], [2, 1], [False, 2], [0, 1.5], [0]])
def test_invalid_sectors_rejected(bounds):
    with pytest.raises(ValueError):
        LayerSpan.from_list(bounds)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 2.5])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        replace(demo_recipe(), seed=seed).validate()


def test_recipe_typos_and_schema_rejected():
    data = demo_recipe().to_dict()
    data["lora_rank"] = 8
    with pytest.raises(ValueError, match="exactly"):
        Recipe.from_dict(data)
    data.pop("lora_rank")
    data["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        Recipe.from_dict(data)


def test_actual_architecture_and_depth_are_checked():
    recipe = demo_recipe()
    mamba = tiny_parent("mamba", 1).config
    qwen = tiny_parent("qwen2", 2).config
    with pytest.raises(ValueError, match="actual architecture"):
        recipe.validate(qwen, mamba)
    with pytest.raises(ValueError, match="exceeds"):
        replace(recipe, donor_sector=LayerSpan(1, 5)).validate(mamba, qwen)
    qwen.use_sliding_window = True
    with pytest.raises(ValueError, match="Sliding-window"):
        recipe.validate(mamba, qwen)


def test_relative_local_paths_resolve_from_recipe_not_cwd(tmp_path):
    data = demo_recipe().to_dict()
    data["recipient"]["local_path"] = "../models/a"
    path = tmp_path / "config" / "recipe.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    assert Recipe.load(path).recipient.local_path == str((tmp_path / "models/a").resolve())
