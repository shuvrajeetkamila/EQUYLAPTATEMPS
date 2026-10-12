"""Complete native block inventories, measured activation statistics, and identity ablations."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from .audit import module_fingerprint
from .backbones import extract_parent
from .recipe import LayerSpan, ParentSpec


def sector_inventory(model, spec: ParentSpec, spans: list[LayerSpan]) -> dict:
    parts = extract_parent(model)
    sectors = []
    for span in spans:
        span.validate(len(parts.layers))
        blocks = torch.nn.ModuleList(parts.layers[span.start : span.stop])
        tensors = [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "parameters": tensor.numel(),
            }
            for name, tensor in blocks.state_dict().items()
        ]
        sectors.append(
            {
                "sector_id": f"{spec.identity.family}:{span.start}:{span.stop}",
                "block_boundaries": [span.start, span.stop],
                "tensor_shapes": tensors,
                "parameter_count": sum(p.numel() for p in blocks.parameters()),
                "sector_sha256": module_fingerprint(blocks),
                "dependencies": (
                    "native RMSNorm, causal convolution, selective SSM; no cached state"
                    if spec.identity.family == "mamba"
                    else "native RMSNorm/MLP/GQA, config-derived RoPE, causal key-padding mask"
                ),
                "candidate_capabilities": ["sequence/language hypothesis"]
                if spec.identity.family == "mamba"
                else ["mathematical-reasoning hypothesis"],
                "evaluation_method": "development activations and identity-replacement NLL",
                "baseline_performance": None,
                "ablation_performance": None,
                "activation_statistics": None,
                "compatibility_status": "PASS",
                "selection_status": "NOT_RUN",
                "selection_reason": "Compatible structure, no capability selection yet",
                "limitations": "Distributed features, not a math organ; token semantics differ.",
            }
        )
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "parent_kind": "synthetic" if spec.identity.synthetic else "pretrained",
        "architecture_family": spec.identity.family,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "hidden_size": model.config.hidden_size,
        "layer_count": len(parts.layers),
        "vocab_size": model.config.vocab_size,
        "sectors": sectors,
    }


@contextmanager
def identity_sector(model, span: LayerSpan):
    """Remove a sector's EFFECT via output hooks. Native compute still occurs; no cache is used."""
    layers = extract_parent(model).layers
    span.validate(len(layers))

    def replace_output(_module, args, output):
        hidden = args[0]
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    hooks = [
        layer.register_forward_hook(replace_output) for layer in layers[span.start : span.stop]
    ]
    try:
        yield model
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def activation_statistics(model, input_batches: list[dict]) -> dict:
    """Scalar aggregates only; never retain a parent's activation targets for seam training."""
    layers = extract_parent(model).layers
    accumulators = [
        {
            "elements": 0,
            "input_sum": 0.0,
            "input_squares": 0.0,
            "output_sum": 0.0,
            "output_squares": 0.0,
            "delta_squares": 0.0,
            "nonfinite": 0,
            "calls": 0,
        }
        for _ in layers
    ]

    def measure(index, _module, args, output):
        x = args[0].detach().float()
        y = (output[0] if isinstance(output, tuple) else output).detach().float()
        if x.shape != y.shape:
            raise ValueError("A complete block changed its residual-stream shape")
        record = accumulators[index]
        record["calls"] += 1
        record["elements"] += x.numel()
        record["input_sum"] += float(x.sum())
        record["input_squares"] += float(x.square().sum())
        record["output_sum"] += float(y.sum())
        record["output_squares"] += float(y.square().sum())
        record["delta_squares"] += float((y - x).square().sum())
        record["nonfinite"] += int((~torch.isfinite(x)).sum() + (~torch.isfinite(y)).sum())

    hooks = [
        layer.register_forward_hook(lambda module, args, out, i=i: measure(i, module, args, out))
        for i, layer in enumerate(layers)
    ]
    started = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        for batch in input_batches:
            model(**batch, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    for record in accumulators:
        count = record["elements"]
        if not count or record["nonfinite"]:
            raise ValueError("Missing/nonfinite block activation observations")
        record["input_mean"] = record["input_sum"] / count
        record["output_mean"] = record["output_sum"] / count
        record["input_rms"] = (record["input_squares"] / count) ** 0.5
        record["output_rms"] = (record["output_squares"] / count) ** 0.5
        record["residual_change_rms"] = (record["delta_squares"] / count) ** 0.5
    return {
        "status": "PASS",
        "kind": "measured",
        "blocks": accumulators,
        "elapsed_seconds": time.monotonic() - started,
        "padding_policy": "unpadded diagnostic inputs",
    }


@contextmanager
def trace_child(child):
    """A default child must call every retained native block in its one integrated path."""
    calls = {role: [0] * len(getattr(child, role).layers) for role in ("prefix", "graft", "suffix")}
    hooks = []
    for role in calls:
        for index, layer in enumerate(getattr(child, role).layers):

            def count(_module, _args, _output, role=role, index=index):
                calls[role][index] += 1

            hooks.append(layer.register_forward_hook(count))
    try:
        yield calls
    finally:
        for hook in hooks:
            hook.remove()


def annotate_selection(inventory: dict, search: dict) -> dict:
    """Attach actual development selection decisions; never infer capability from block names."""
    import copy
    import math

    result = copy.deepcopy(inventory)
    candidates = {tuple(candidate["span"]): candidate for candidate in search["candidates"]}
    selected = tuple(search["selected"])
    available = {tuple(sector["block_boundaries"]) for sector in result["sectors"]}
    if selected not in available or selected not in candidates:
        raise ValueError("Selected sector is absent from the measured inventory/search")
    completed = [candidate for candidate in search["candidates"] if candidate["status"] == "PASS"]
    if not completed or any(not math.isfinite(c["development_selection_score"]) for c in completed):
        raise ValueError("Selection needs finite completed candidate scores")
    winner = min(completed, key=lambda c: c["development_selection_score"])
    if tuple(winner["span"]) != selected:
        raise ValueError("Selected sector violates the registered lowest-development-score rule")
    for sector in result["sectors"]:
        span = tuple(sector["block_boundaries"])
        if span not in candidates:
            sector["selection_status"] = "NOT_RUN"
            sector["selection_reason"] = "Compatible but not calibrated in this bounded search"
            continue
        candidate = candidates[span]
        score = candidate["development_selection_score"]
        if candidate["status"] != "PASS" or not math.isfinite(score):
            raise ValueError("Selection needs a finite completed development observation")
        sector["selection_status"] = "SELECTED" if span == selected else "REJECTED"
        sector["selection_reason"] = search["selection_reason"]
        sector["development_selection_score"] = score
        sector["selection_interpretation"] = (
            "Development-fit selection, not isolated capability proof"
        )
    return result
