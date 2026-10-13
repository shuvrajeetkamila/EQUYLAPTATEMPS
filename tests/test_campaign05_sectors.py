"""Campaign 5 sector-width mechanics: variable donor spans, twin parity."""

import torch

from equylaptatemps.controls import random_qwen_sector_control
from equylaptatemps.model import child_from_parents
from equylaptatemps.recipe import LayerSpan, Recipe
from equylaptatemps.sequence_seams import install_bottleneck_interfaces
from equylaptatemps.synthetic import demo_recipe, tiny_parent

BN_DELTA = {
    "family": "bottleneck",
    "kind": "causal_attention",
    "bottleneck": 8,
    "heads": 2,
    "max_positions": 32,
    "donor_delta": True,
}


def _child(sector, random_twin=False):
    base = demo_recipe()
    recipe = Recipe(base.recipient, base.donor, LayerSpan(1, 3), LayerSpan(*sector), 17)
    recipient, donor = tiny_parent("mamba", 117), tiny_parent("qwen2", 217)
    if random_twin:
        child = random_qwen_sector_control(recipe, recipient, donor, initialization_seed=2024)
    else:
        child = child_from_parents(recipe, recipient, donor)
    return install_bottleneck_interfaces(child, BN_DELTA)


def test_variable_sector_widths_execute_and_match_twin_parameters():
    for sector in [(1, 2), (1, 3), (0, 4)]:
        pretrained, twin = _child(sector), _child(sector, random_twin=True)
        assert len(pretrained.graft.layers) == sector[1] - sector[0]
        def trainable(c):
            return sum(p.numel() for p in c.parameters() if p.requires_grad)
        assert trainable(pretrained) == trainable(twin)
        ids = torch.tensor([[8, 9, 10, 11]])
        with torch.inference_mode():
            assert torch.isfinite(pretrained(ids).logits).all()
            assert torch.isfinite(twin(ids).logits).all()
            assert not torch.allclose(
                pretrained(ids).logits, pretrained(ids, ablation="bypass_graft").logits
            )
