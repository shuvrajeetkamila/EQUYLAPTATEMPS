"""Explicit evaluation controls; random grafts are never claimed as pretrained transplantation."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import config_from_dict, make_block
from .model import child_from_parents


def random_qwen_sector_control(recipe, recipient, donor, *, initialization_seed: int):
    """Same shape/native operations, fresh frozen donor block parameters, same seam seed.

    Uses the pinned Transformers model's standard initialization on only the chosen
    blocks, not a second gigabyte-sized random parent. Parent modules are not altered.
    Only interface parameters may train, with exactly the real child's training budget.
    Supports Qwen2 (Campaigns 2-6) and Llama (Campaign 7) donor sectors.
    """
    if donor.config.model_type not in ("qwen2", "llama"):
        raise ValueError("This matched control supports native Qwen2/Llama donor sectors only")
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


# Campaign 7 alias: the control is family-general (Qwen2 and Llama sectors).
random_sector_control = random_qwen_sector_control


def graft_activity(child, tokenizer, texts) -> dict:
    """Mean relative graft update ||out-in||/||in|| at the child's current state.

    Label-free; measured on provided (TRAIN) texts only. Used by the Campaign
    10 awake-control calibration and diagnostics.
    """
    captured = {}

    def hook(_module, args, output):
        captured["input"] = args[0].detach()
        captured["output"] = output.detach()
        return None

    handle = child.graft.register_forward_hook(hook)
    ratios = []
    try:
        with torch.inference_mode():
            for text in texts:
                ids = torch.tensor(
                    [tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long
                )
                child(ids)
                delta = captured["output"] - captured["input"]
                ratios.append(float(delta.norm() / captured["input"].norm().clamp_min(1e-9)))
    finally:
        handle.remove()
    finite = all(torch.isfinite(torch.tensor(r)) for r in ratios) if ratios else False
    return {
        "mean_relative_update": sum(ratios) / len(ratios) if ratios else None,
        "texts": len(ratios),
        "all_finite": bool(finite),
    }


def native_sector_activity(donor, donor_tokenizer, texts, sector) -> dict:
    """The donor sector's own relative update inside its native model.

    Captures the residual stream entering and leaving the sector while the
    frozen donor runs natively. This is Campaign 10's label-free activity
    target for the awake random twin.
    """
    from .backbones import extract_parent

    layers = extract_parent(donor).layers
    captured = {}

    def make_hook(name):
        def hook(_module, _args, output):
            captured[name] = (output[0] if isinstance(output, tuple) else output).detach()
            return None

        return hook

    h_in = layers[sector.start - 1].register_forward_hook(make_hook("sector_in"))
    h_out = layers[sector.stop - 1].register_forward_hook(make_hook("sector_out"))
    ratios = []
    try:
        with torch.inference_mode():
            for text in texts:
                ids = torch.tensor(
                    [donor_tokenizer.encode(text, add_special_tokens=False)],
                    dtype=torch.long,
                )
                donor(ids, use_cache=False)
                delta = captured["sector_out"] - captured["sector_in"]
                ratios.append(float(delta.norm() / captured["sector_in"].norm().clamp_min(1e-9)))
    finally:
        h_in.remove()
        h_out.remove()
    return {
        "mean_relative_update": sum(ratios) / len(ratios) if ratios else None,
        "texts": len(ratios),
        "sector": [sector.start, sector.stop],
    }


def calibrate_awake_sector(
    child, tokenizer, texts, *, target: float, bounds=(0.05, 1000.0), band=(0.5, 2.0)
) -> dict:
    """Scale a random graft's residual-output projections to hit an activity target.

    Frozen rule (campaigns/campaign_10/awake_control_design.md): one scalar on
    every graft block's o_proj/down_proj weights, estimated from the measured
    relative update, clipped to `bounds`, one correction allowed. Successful
    only if the achieved activity lands within `band` x target, all finite.
    The scaled weights remain frozen; nothing here trains.
    """
    import math

    def apply(scale: float) -> None:
        with torch.no_grad():
            for layer in child.graft.layers:
                layer.self_attn.o_proj.weight.mul_(scale)
                layer.mlp.down_proj.weight.mul_(scale)

    record = {"target_relative_update": target, "calibration_texts": len(texts)}
    initial = graft_activity(child, tokenizer, texts)
    record["initial"] = initial
    u0 = initial["mean_relative_update"]
    if not initial["all_finite"] or not u0 or not math.isfinite(u0) or u0 <= 0:
        record.update(calibrated=False, reason="initial activity not measurable/finite")
        return record
    # The TOTAL applied scalar is what the frozen bounds govern (Campaign 10
    # Amendment 01: an earlier version clipped each incremental step, letting
    # the total exceed the bound). Two attempts: estimate, then one correction.
    total = 1.0
    achieved = initial
    for _attempt in range(2):
        desired_total = total * (target / achieved["mean_relative_update"])
        desired_total = min(max(desired_total, bounds[0]), bounds[1])
        apply(desired_total / total)
        total = desired_total
        achieved = graft_activity(child, tokenizer, texts)
        record["achieved"] = achieved
        u = achieved["mean_relative_update"]
        if achieved["all_finite"] and u is not None and band[0] * target <= u <= band[1] * target:
            record.update(calibrated=True, scale_applied=total)
            return record
        if not achieved["all_finite"] or not u or u <= 0:
            break
    record.update(calibrated=False, scale_applied=total, reason="activity outside frozen band")
    return record
