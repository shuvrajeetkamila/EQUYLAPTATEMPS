"""Request-local native KV/SSM incremental decoding, with the cache-free path kept as oracle.

Initial supported scope is CPU float32, batch1 without padding. No weights, block
indices, config or model architecture are changed. Caches are activation state only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.mamba.modeling_mamba import MambaCache

from .model import SurgeryOutput, SurgicalChild

ABLATIONS = {
    "none",
    "bypass_graft",
    "zero_graft",
    "bypass_prefix",
    "bypass_suffix",
    "zero_input_seam",
    "zero_output_seam",
}


def new_native_cache(config, *, device, dtype):
    if config.model_type == "mamba":
        if config.conv_kernel < 2:
            raise ValueError("Cached Mamba kernels shorter than2 are not validated")
        return MambaCache(config, max_batch_size=1, device=device, dtype=dtype)
    if config.model_type == "qwen2":
        return DynamicCache()
    raise ValueError("Only native Mamba and Qwen2 caching is supported")


def cached_segment(segment, hidden, cache, *, start: int, prefill: bool):
    if not segment.layers:
        return hidden
    length = hidden.shape[1]
    if segment.config.model_type == "mamba":
        positions = (
            torch.arange(segment.config.conv_kernel, device=hidden.device)
            if prefill
            else torch.tensor([segment.config.conv_kernel], device=hidden.device)
        )
        for layer in segment.layers:
            hidden = layer(
                hidden, cache_params=cache, cache_position=positions, attention_mask=None
            )
        return hidden
    positions = torch.arange(start, start + length, device=hidden.device)
    position_ids = positions.unsqueeze(0)
    key_positions = torch.arange(start + length, device=hidden.device)
    blocked = key_positions[None, :] > positions[:, None]
    mask = torch.zeros(1, 1, length, start + length, device=hidden.device, dtype=hidden.dtype)
    mask = mask.masked_fill(blocked[None, None], torch.finfo(hidden.dtype).min)
    embeddings = segment.rotary(hidden, position_ids)
    for layer in segment.layers:
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=positions,
            position_embeddings=embeddings,
        )[0]
    return hidden


class CachedDecoder:
    """One request for one unchanged model/control. Never share state between prompts."""

    def __init__(self, model, *, ablation: str = "none"):
        if ablation not in ABLATIONS:
            raise ValueError("Unknown cached evaluation ablation")
        if not isinstance(model, SurgicalChild) and ablation != "none":
            raise ValueError("Native-parent baselines cannot use child ablations")
        parameter = next(model.parameters())
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("Initial cached validation supports CPU float32 only")
        if model.training:
            raise ValueError("Incremental caches are inference-only; use eval mode")
        self._model = model
        self._ablation = ablation
        self._parameter_versions = tuple(parameter._version for parameter in model.parameters())
        self.length = 0
        self.started = False
        self.device = parameter.device
        self.dtype = parameter.dtype
        if isinstance(model, SurgicalChild):
            self.recipient_cache = new_native_cache(
                model.recipient_config, device=self.device, dtype=self.dtype
            )
            self.donor_cache = new_native_cache(
                model.donor_config, device=self.device, dtype=self.dtype
            )
        else:
            self.native_cache = new_native_cache(model.config, device=self.device, dtype=self.dtype)

    @property
    def model(self):
        return self._model

    @property
    def ablation(self):
        return self._ablation

    def _check(self, ids, prefill):
        if (
            tuple(parameter._version for parameter in self.model.parameters())
            != self._parameter_versions
        ):
            raise ValueError("Model parameters changed inside a cached request")
        if self.model.training:
            raise ValueError("Cached request cannot change to training mode")
        if (
            next(self.model.parameters()).device != self.device
            or next(self.model.parameters()).dtype != self.dtype
        ):
            raise ValueError("Cached request cannot move its model device/dtype")
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[1] < 1
            or ids.dtype != torch.long
            or ids.device != self.device
        ):
            raise ValueError("Cached input must be unpadded batch1 CPU int64")
        if prefill == self.started:
            raise ValueError("Prefill exactly once; decode only after prefill")
        if not prefill and ids.shape[1] != 1:
            raise ValueError("Incremental decode accepts exactly one new token")
        config = (
            self.model.recipient_config
            if isinstance(self.model, SurgicalChild)
            else self.model.config
        )
        if torch.any(ids < 0) or torch.any(ids >= config.vocab_size):
            raise ValueError("Cached input IDs must use the original recipient vocabulary")
        limit = (
            self.model.context_limit
            if isinstance(self.model, SurgicalChild)
            else getattr(config, "max_position_embeddings", None)
        )
        if limit is not None and self.length + ids.shape[1] > limit:
            raise ValueError("Cached sequence exceeds native/transplanted context budget")

    @torch.inference_mode()
    def _forward(self, ids, *, prefill):
        self._check(ids, prefill)
        start = self.length
        model = self.model
        if isinstance(model, SurgicalChild):
            hidden = model.embedding(ids)
            if self.ablation != "bypass_prefix":
                hidden = cached_segment(
                    model.prefix, hidden, self.recipient_cache, start=start, prefill=prefill
                )
            hidden = model.input_seam(hidden)
            if self.ablation == "zero_input_seam":
                hidden = torch.zeros_like(hidden)
            if self.ablation != "bypass_graft":
                hidden = cached_segment(
                    model.graft, hidden, self.donor_cache, start=start, prefill=prefill
                )
            if self.ablation == "zero_graft":
                hidden = torch.zeros_like(hidden)
            hidden = model.output_seam(hidden)
            if self.ablation == "zero_output_seam":
                hidden = torch.zeros_like(hidden)
            if self.ablation != "bypass_suffix":
                hidden = cached_segment(
                    model.suffix, hidden, self.recipient_cache, start=start, prefill=prefill
                )
            hidden = model.final_norm(hidden).to(model.lm_head.weight.dtype)
            logits = model.lm_head(hidden).float()
        elif model.config.model_type == "mamba":
            positions = (
                torch.arange(model.config.conv_kernel, device=self.device)
                if prefill
                else torch.tensor([model.config.conv_kernel], device=self.device)
            )
            output = model(
                ids, use_cache=True, cache_params=self.native_cache, cache_position=positions
            )
            self.native_cache = output.cache_params
            logits = output.logits
        else:
            positions = torch.arange(start, start + ids.shape[1], device=self.device)
            output = model(
                ids,
                use_cache=True,
                past_key_values=self.native_cache,
                attention_mask=torch.ones(
                    1, start + ids.shape[1], dtype=torch.long, device=self.device
                ),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
            )
            self.native_cache = output.past_key_values
            logits = output.logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite cached model logits")
        self.length += ids.shape[1]
        self.started = True
        return SurgeryOutput(logits.float(), None)

    def prefill(self, ids):
        return self._forward(ids, prefill=True)

    def step(self, ids):
        return self._forward(ids, prefill=False)


@dataclass(frozen=True)
class Generation:
    text: str
    generated_ids: list[int]
    stop_reason: str
    input_tokens: int


@torch.inference_mode()
def cached_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    ablation: str = "none",
    stop_when=None,
) -> Generation:
    if max_new_tokens < 1 or max_input_tokens < 1 or not prompt.strip():
        raise ValueError("Generation needs a prompt and positive input/output token budgets")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_input_tokens:
        raise ValueError("Do not silently truncate a registered prompt")
    if (
        isinstance(model, SurgicalChild)
        and model.context_limit is not None
        and len(ids) + max_new_tokens > model.context_limit
    ):
        raise ValueError("Prompt/output budget exceeds integrated context limit")
    decoder = CachedDecoder(model, ablation=ablation)
    logits = decoder.prefill(torch.tensor([ids], dtype=torch.long)).logits
    tokens = []
    reason = "max_new_tokens"
    for index in range(max_new_tokens):
        token = int(logits[0, -1].argmax())
        if token == tokenizer.eos_token_id:
            reason = "eos"
            break
        tokens.append(token)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if stop_when is not None and stop_when(text):
            reason = "completed_answer"
            break
        if index + 1 < max_new_tokens:
            logits = decoder.step(torch.tensor([[token]], dtype=torch.long)).logits
    return Generation(tokenizer.decode(tokens, skip_special_tokens=True), tokens, reason, len(ids))
