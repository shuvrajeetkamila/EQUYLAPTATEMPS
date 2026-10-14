"""Local offline smoke: synthetic Mamba host -> synthetic Llama donor sector."""

import torch

from equylaptatemps.controls import random_sector_control
from equylaptatemps.model import child_from_parents, empty_child
from equylaptatemps.recipe import LayerSpan, ParentSpec, Recipe
from equylaptatemps.synthetic import tiny_parent


def main() -> None:
    recipe = Recipe(
        ParentSpec("synthetic://mamba", None, None),
        ParentSpec("synthetic://llama", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        42,
    )
    recipient = tiny_parent("mamba", 1)
    donor = tiny_parent("llama", 2)
    child = child_from_parents(recipe, recipient, donor)
    print("structure:", child.structure_report())
    ids = torch.randint(3, 250, (2, 12))
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :4] = -100
    child.train(True)
    out = child(ids, mask, labels)
    print("loss:", float(out.loss))
    out.loss.backward()
    grads = sorted(
        {
            n.split(".")[0]
            for n, p in child.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
    )
    print("params with grad:", grads)
    child.eval()
    for ab in ("none", "bypass_graft", "zero_graft"):
        print(ab, round(float(child(ids, mask, labels, ablation=ab).loss), 4))
    twin = random_sector_control(recipe, recipient, donor, initialization_seed=1234)
    print("twin loss:", round(float(twin(ids, mask, labels).loss), 4))
    ec = empty_child(recipe, recipient.config.to_dict(), donor.config.to_dict())
    print("empty_child loss:", round(float(ec(ids, mask, labels).loss), 4))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
