"""Explicit evaluation controls; random grafts are never claimed as pretrained transplantation."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import config_from_dict, make_block
from .model import child_from_parents


def random_qwen_sector_control(recipe, recipient, donor, *, initialization_seed: int):
    """Same shape/native operations, fresh frozen Qwen block parameters, same new seam seed.

    Uses the pinned Transformers model's standard initialization on only the chosen
    blocks, not a second gigabyte-sized random parent. Parent modules are not altered.
    Only interface parameters may train, with exactly the real child's training budget.
    """
    if donor.config.model_type != "qwen2":
        raise ValueError("This matched control supports native Qwen donor sectors only")
    child = child_from_parents(recipe, recipient, donor)
    config = config_from_dict(donor.config.to_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        blocks = nn.ModuleList(
            [
                make_block(config, i)
                for i in range(recipe.donor_sector.start, recipe.donor_sector.stop)
            ]
        )
        blocks.apply(donor._init_weights)
    blocks.to(device=child.device, dtype=child.embedding.weight.dtype).requires_grad_(False).eval()
    child.graft.layers = blocks
    child.random_graft_control = True
    child.assert_trainable_contract()
    return child.eval()
