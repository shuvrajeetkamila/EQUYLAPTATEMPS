"""Portable execution contracts: tensor hashes alone do not protect native operator semantics."""

from __future__ import annotations

import hashlib
import json

# Provenance/installation metadata, not a numerical execution setting.
LOCATION_KEYS = {"_name_or_path", "_commit_hash", "transformers_version"}
OPERATOR_ATTRIBUTES = (
    "eps",
    "epsilon",
    "variance_epsilon",
    "layer_idx",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "scaling",
    "attention_dropout",
    "conv_kernel_size",
    "ssm_state_size",
    "time_step_rank",
    "use_bias",
    "use_conv_bias",
    "residual_in_fp32",
    "padding_idx",
    "normalized_shape",
    "p",
)


def canonical_config(config) -> dict:
    values = {key: value for key, value in config.to_dict().items() if key not in LOCATION_KEYS}
    return json.loads(json.dumps(values, sort_keys=True, allow_nan=False))


def native_operations(module) -> dict:
    result = {}
    for name, component in module.named_modules():
        attributes = {}
        for attribute in OPERATOR_ATTRIBUTES:
            value = getattr(component, attribute, None)
            if isinstance(value, str | int | float | bool) or (
                isinstance(value, list | tuple)
                and all(isinstance(v, str | int | float | bool) for v in value)
            ):
                attributes[attribute] = list(value) if isinstance(value, tuple) else value
        result[name] = {
            "class": type(component).__module__ + "." + type(component).__qualname__,
            "attributes": attributes,
        }
    return result


def parent_contract(model) -> dict:
    return {"config": canonical_config(model.config), "operations": native_operations(model)}


def child_contract(child) -> dict:
    recipe = child.recipe.to_dict()
    for role in ("recipient", "donor"):
        recipe[role].pop("local_path", None)
    return {
        "recipe": recipe,
        "recipient_config": canonical_config(child.recipient_config),
        "donor_config": canonical_config(child.donor_config),
        "operations": native_operations(child),
    }


def contract_sha256(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
