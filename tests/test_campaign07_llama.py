"""Campaign 7 Llama-family support: synthetic, offline, no pretrained weights."""

import torch

from equylaptatemps.backbones import config_from_dict, extract_parent, make_block, make_norm
from equylaptatemps.controls import random_sector_control
from equylaptatemps.model import child_from_parents, empty_child
from equylaptatemps.recipe import REGISTRY, LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent

SMOLLM_CONFIG = {
    "model_type": "llama",
    "hidden_size": 576,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "intermediate_size": 1536,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "tie_word_embeddings": True,
    "vocab_size": 49152,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
}


def _recipe():
    return Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )


def _child():
    recipe = _recipe()
    return recipe, child_from_parents(recipe, tiny_parent("mamba", 1), tiny_parent("llama", 2))


def _batch():
    ids = torch.randint(3, 250, (2, 12))
    labels = ids.clone()
    labels[:, :4] = -100
    return ids, torch.ones_like(ids), labels


def test_smollm_config_loads_as_llama():
    config = config_from_dict(SMOLLM_CONFIG)
    assert config.model_type == "llama"
    assert config.hidden_size == 576 and config.num_hidden_layers == 30
    assert config.num_key_value_heads == 3


def test_provenance_registry_has_reviewed_smollm():
    identity = REGISTRY["HuggingFaceTB/SmolLM-135M"]
    assert identity.family == "llama" and not identity.synthetic
    spec = ParentSpec(
        "HuggingFaceTB/SmolLM-135M", "/tmp/x", "1d461723eec654e65efdc40cf49301c89c0c92f4"
    )
    spec.validate()


def test_llama_blocks_and_norm_construct():
    config = config_from_dict(SMOLLM_CONFIG)
    block = make_block(config, 24)
    assert type(block).__name__ == "LlamaDecoderLayer"
    assert block.self_attn.q_proj.in_features == 576
    assert block.self_attn.k_proj.out_features == 192  # GQA: 3 KV heads x 64
    assert type(make_norm(config)).__name__ == "LlamaRMSNorm"


def test_synthetic_llama_parent_extracts():
    parts = extract_parent(tiny_parent("llama", 3))
    assert len(parts.layers) == 4
    assert parts.embedding.weight.shape[1] == 24


def test_mamba_llama_child_forward_backward_and_ablations():
    _, child = _child()
    ids, mask, labels = _batch()
    report = child.structure_report()
    assert report["recipient_family"] == "mamba" and report["donor_family"] == "llama"
    assert report["context_limit"] == 256
    child.train(True)
    loss = child(ids, mask, labels).loss
    loss.backward()
    trainable = {n.split(".")[0] for n, p in child.named_parameters() if p.requires_grad}
    assert trainable == {"input_seam", "output_seam"}
    child.eval()
    for ablation in ("none", "bypass_graft", "zero_graft"):
        assert torch.isfinite(child(ids, mask, labels, ablation=ablation).loss)


def test_llama_random_sector_control_is_matched_and_frozen():
    recipe, child = _child()
    twin = random_sector_control(
        recipe, tiny_parent("mamba", 1), tiny_parent("llama", 2), initialization_seed=1234
    )
    ids, mask, labels = _batch()
    real_shapes = [tuple(p.shape) for p in child.graft.parameters()]
    twin_shapes = [tuple(p.shape) for p in twin.graft.parameters()]
    assert real_shapes == twin_shapes
    assert not any(p.requires_grad for p in twin.graft.parameters())
    assert torch.isfinite(twin(ids, mask, labels).loss)
    twin.assert_trainable_contract()


def test_empty_child_roundtrip_llama():
    recipe = _recipe()
    ec = empty_child(
        recipe, tiny_parent("mamba", 1).config.to_dict(), tiny_parent("llama", 2).config.to_dict()
    )
    ids, mask, labels = _batch()
    assert torch.isfinite(ec(ids, mask, labels).loss)
