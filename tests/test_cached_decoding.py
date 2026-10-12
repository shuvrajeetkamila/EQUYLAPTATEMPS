"""Native cache/full-prefix equivalence: offline fixtures are not capability evidence."""

from dataclasses import replace

import pytest
import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.decoding import CachedDecoder, cached_generate
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan
from equylaptatemps.synthetic import demo_recipe, tiny_parent


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "ablation",
    [
        "none",
        "bypass_graft",
        "zero_graft",
        "bypass_prefix",
        "bypass_suffix",
        "zero_output_seam",
    ],
)
def test_incremental_native_child_equals_uncached_both_directions(reverse, ablation):
    recipe = demo_recipe(reverse=reverse)
    a = tiny_parent(recipe.recipient.identity.family, 117)
    b = tiny_parent(recipe.donor.identity.family, 217)
    child = child_from_parents(recipe, a, b)
    before = module_fingerprint(child)
    ids = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]])
    decoder = CachedDecoder(child, ablation=ablation)
    with torch.inference_mode():
        pref = decoder.prefill(ids[:, :3]).logits
        full = child(ids[:, :3], ablation=ablation).logits
        torch.testing.assert_close(pref, full, atol=2e-5, rtol=2e-5)
        for i in range(3, ids.shape[1]):
            cached = decoder.step(ids[:, i : i + 1]).logits[:, -1]
            oracle = child(ids[:, : i + 1], ablation=ablation).logits[:, -1]
            torch.testing.assert_close(cached, oracle, atol=2e-5, rtol=2e-5)
    assert decoder.length == ids.shape[1]
    assert module_fingerprint(child) == before


@pytest.mark.parametrize("family", ["mamba", "qwen2"])
def test_native_parent_caches_match_same_full_model(family):
    parent = tiny_parent(family, 117)
    ids = torch.tensor([[8, 9, 10, 11, 12]])
    decoder = CachedDecoder(parent)
    with torch.inference_mode():
        decoder.prefill(ids[:, :2])
        for i in range(2, ids.shape[1]):
            cached = decoder.step(ids[:, i : i + 1]).logits[:, -1]
            full = parent(ids[:, : i + 1], use_cache=False).logits[:, -1]
            torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_original_indices_and_empty_prefix_suffix(reverse):
    recipe = demo_recipe(reverse=reverse)
    for cut in (LayerSpan(0, 2), LayerSpan(2, 4)):
        r = replace(recipe, recipient_cut=cut, donor_sector=LayerSpan(2, 4))
        a = tiny_parent(r.recipient.identity.family, 117)
        b = tiny_parent(r.donor.identity.family, 217)
        child = child_from_parents(r, a, b)
        decoder = CachedDecoder(child)
        with torch.inference_mode():
            decoder.prefill(torch.tensor([[8, 9]]))
            actual = decoder.step(torch.tensor([[10]])).logits[:, -1]
            expected = child(torch.tensor([[8, 9, 10]])).logits[:, -1]
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_request_isolation_and_bad_input_are_explicit(child_bundle):
    child, *_ = child_bundle
    first = CachedDecoder(child)
    second = CachedDecoder(child)
    assert first.recipient_cache is not second.recipient_cache
    with pytest.raises(ValueError, match="after prefill"):
        first.step(torch.tensor([[8]]))
    with pytest.raises(ValueError, match="batch1"):
        first.prefill(torch.tensor([[8], [9]]))
    first.prefill(torch.tensor([[8, 9]]))
    with pytest.raises(ValueError, match="Prefill exactly once"):
        first.prefill(torch.tensor([[8]]))
    with pytest.raises(ValueError, match="one new token"):
        first.step(torch.tensor([[8, 9]]))
    child.train()
    with pytest.raises(ValueError, match="inference-only"):
        CachedDecoder(child)


def test_cached_generation_records_stopping_not_fake_capability(child_bundle):
    child, tok, *_ = child_bundle
    result = cached_generate(child, tok, "Unit fixture.", max_new_tokens=4, max_input_tokens=64)
    assert result.stop_reason in {"eos", "max_new_tokens"}
    assert result.input_tokens > 0
    assert all(0 <= i < child.recipient_config.vocab_size for i in result.generated_ids)
    with pytest.raises(ValueError, match="positive"):
        cached_generate(child, tok, "X", max_new_tokens=0, max_input_tokens=64)


def test_unvalidated_real_zero_input_control_is_explicitly_blocked(child_bundle):
    child, *_ = child_bundle
    with pytest.raises(ValueError, match="failed real-weight"):
        CachedDecoder(child, ablation="zero_input_seam")
    # Original uncached control remains intact and independently executable.
    with torch.inference_mode():
        assert torch.isfinite(
            child(torch.tensor([[8, 9, 10]]), ablation="zero_input_seam").logits
        ).all()
