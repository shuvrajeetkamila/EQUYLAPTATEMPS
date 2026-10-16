"""Pinned, cache-free execution of intact native blocks. No remote model code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig, MambaConfig, Qwen2Config
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.mamba.modeling_mamba import MambaBlock, MambaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

_ROTARY = {"qwen2": Qwen2RotaryEmbedding, "llama": LlamaRotaryEmbedding}


def config_from_dict(data: dict):
    family = data.get("model_type")
    classes = {"mamba": MambaConfig, "qwen2": Qwen2Config, "llama": LlamaConfig}
    if family not in classes:
        raise ValueError(f"Unsupported architecture {family!r}")
    config = classes[family].from_dict(data)
    config.use_cache = False
    config._attn_implementation = "eager"
    if family == "qwen2" and config.use_sliding_window:
        raise ValueError("Sliding-window attention is not implemented")
    if family == "llama" and getattr(config, "sliding_window", None) is not None:
        raise ValueError("Sliding-window attention is not implemented")
    return config


@dataclass
class ParentParts:
    config: object
    embedding: nn.Embedding
    layers: list[nn.Module]
    norm: nn.Module
    head: nn.Linear


def extract_parent(parent: nn.Module) -> ParentParts:
    if parent.config.model_type == "mamba":
        return ParentParts(
            parent.config,
            parent.backbone.embeddings,
            list(parent.backbone.layers),
            parent.backbone.norm_f,
            parent.lm_head,
        )
    if parent.config.model_type == "qwen2":
        if any(type(layer.self_attn).__name__ != "Qwen2Attention" for layer in parent.model.layers):
            raise ValueError("Load Qwen2 with attn_implementation='eager' for this pinned runtime")
        return ParentParts(
            parent.config,
            parent.model.embed_tokens,
            list(parent.model.layers),
            parent.model.norm,
            parent.lm_head,
        )
    if parent.config.model_type == "llama":
        if any(type(layer.self_attn).__name__ != "LlamaAttention" for layer in parent.model.layers):
            raise ValueError("Load Llama with attn_implementation='eager' for this pinned runtime")
        return ParentParts(
            parent.config,
            parent.model.embed_tokens,
            list(parent.model.layers),
            parent.model.norm,
            parent.lm_head,
        )
    raise ValueError(f"Unsupported architecture {parent.config.model_type!r}")


def make_block(config, index: int) -> nn.Module:
    if config.model_type == "mamba":
        return MambaBlock(config, index)
    if config.model_type == "llama":
        return LlamaDecoderLayer(config, index)
    return Qwen2DecoderLayer(config, index)


def make_norm(config) -> nn.Module:
    if config.model_type == "mamba":
        return MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    if config.model_type == "llama":
        return LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def causal_mask(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Additive [B,1,T,T] mask: right padding and the future are never visible."""
    batch, length = attention_mask.shape
    future = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
    blocked = future[None, None, :, :] | ~attention_mask[:, None, None, :].bool()
    result = torch.zeros(batch, 1, length, length, device=hidden.device, dtype=hidden.dtype)
    return result.masked_fill(blocked, torch.finfo(hidden.dtype).min)


class StackSegment(nn.Module):
    """A contiguous native sector with its own positional/state-space machinery."""

    def __init__(self, config, layers: list[nn.Module]):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(layers)
        device = next(self.layers.parameters()).device if layers else None
        rotary_cls = _ROTARY.get(config.model_type)
        self.rotary = rotary_cls(config, device=device) if rotary_cls is not None else None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.rotary is not None:
            # RoPE frequencies are nonpersistent and derived from config. Recreate in fp32,
            # rather than irreversibly rounding them when .to(bfloat16) moves the child.
            device = self.rotary.inv_freq.device
            self.rotary = _ROTARY[self.config.model_type](self.config, device=device).train(
                self.training
            )
        return self

    def forward(self, hidden, attention_mask, *, gradient_checkpointing: bool = False):
        if not self.layers:
            return hidden
        if self.config.model_type in ("qwen2", "llama"):
            position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            mask = causal_mask(hidden, attention_mask)
            embeddings = self.rotary(hidden, position_ids)
            for layer in self.layers:
                # Bind the layer now: a late-bound loop closure breaks backward recomputation.
                def run(x, block=layer):
                    return block(
                        x,
                        attention_mask=mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=positions,
                        position_embeddings=embeddings,
                    )[0]

                if gradient_checkpointing and torch.is_grad_enabled() and hidden.requires_grad:
                    hidden = checkpoint(run, hidden, use_reentrant=False)
                else:
                    hidden = run(hidden)
        else:
            for layer in self.layers:
                run = partial(
                    layer,
                    cache_params=None,
                    cache_position=None,
                    attention_mask=attention_mask,
                )
                if gradient_checkpointing and torch.is_grad_enabled() and hidden.requires_grad:
                    hidden = checkpoint(run, hidden, use_reentrant=False)
                else:
                    hidden = run(hidden)
        return hidden
