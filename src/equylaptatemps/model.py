"""A single causal child: recipient prefix -> native donor sector -> recipient suffix."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import (
    StackSegment,
    config_from_dict,
    extract_parent,
    make_block,
    make_norm,
)
from .recipe import Recipe


class DenseSeam(nn.Module):
    """Train the ENTIRE new interface matrix; never a low-rank parent-weight update."""

    def __init__(self, source_width: int, destination_width: int):
        super().__init__()
        self.norm = nn.LayerNorm(source_width)
        self.projection = nn.Linear(source_width, destination_width, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Mamba may deliberately keep its residual stream in fp32 even with bf16 weights.
        hidden = hidden.to(self.projection.weight.dtype)
        return self.projection(self.norm(hidden))


@dataclass
class SurgeryOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class SurgicalChild(nn.Module):
    """No router, ensemble, logit combination, residual bypass, or parent tensor averaging."""

    def __init__(
        self,
        recipe: Recipe,
        recipient_config,
        donor_config,
        embedding: nn.Embedding,
        prefix: StackSegment,
        graft: StackSegment,
        suffix: StackSegment,
        final_norm: nn.Module,
        lm_head: nn.Linear,
    ):
        super().__init__()
        recipe.validate(recipient_config, donor_config)
        self.recipe = recipe
        self.recipient_config = recipient_config
        self.donor_config = donor_config
        self.embedding = embedding
        self.prefix = prefix
        self.graft = graft
        self.suffix = suffix
        self.final_norm = final_norm
        self.lm_head = lm_head
        # Parent modules are frozen, but autograd still traverses the graft and suffix.
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(recipe.seed)
            self.input_seam = DenseSeam(recipient_config.hidden_size, donor_config.hidden_size)
            self.output_seam = DenseSeam(donor_config.hidden_size, recipient_config.hidden_size)
        self.input_seam.to(
            device=embedding.weight.device,
            dtype=graft.layers[0].norm.weight.dtype
            if donor_config.model_type == "mamba"
            else graft.layers[0].input_layernorm.weight.dtype,
        )
        self.output_seam.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
        self.gradient_checkpointing = False
        # Campaign 4 mandatory-use option: the suffix receives only the donor
        # sector's *change* to its input (graft output minus graft input), so
        # an identity donor yields a zero signal. Default off (Campaigns 1-3
        # semantics unchanged); set only by registered interface installs.
        self.donor_delta = False
        self.train(False)

    @property
    def device(self):
        return self.embedding.weight.device

    @property
    def context_limit(self) -> int | None:
        limits = [
            config.max_position_embeddings
            for config in (self.recipient_config, self.donor_config)
            if config.model_type in ("qwen2", "llama")
        ]
        return min(limits) if limits else None

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen means both immutable tensors and deterministic native dropout behavior.
        for module in (
            self.embedding,
            self.prefix,
            self.graft,
            self.suffix,
            self.final_norm,
            self.lm_head,
        ):
            module.eval()
        return self

    def seam_parameters(self):
        return [*self.input_seam.parameters(), *self.output_seam.parameters()]

    def assert_trainable_contract(self) -> None:
        for name, parameter in self.named_parameters():
            expected = name.startswith(("input_seam.", "output_seam."))
            if parameter.requires_grad != expected:
                raise ValueError(f"Trainable/frozen surgery contract broken for {name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        ablation: str = "none",
    ) -> SurgeryOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] int64 tensor")
        if input_ids.min() < 0 or input_ids.max() >= self.recipient_config.vocab_size:
            raise ValueError("Input IDs must use the recipient vocabulary, never donor IDs")
        if self.context_limit is not None and input_ids.shape[1] > self.context_limit:
            raise ValueError(
                f"Sequence exceeds the transplanted context limit {self.context_limit}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask must be binary and match input_ids")
        if not torch.all(attention_mask[:, 0] == 1) or torch.any(
            attention_mask[:, 1:] > attention_mask[:, :-1]
        ):
            raise ValueError("Only nonempty, right-padded sequences are supported")
        if ablation not in {
            "none",
            "bypass_graft",
            "zero_graft",
            "bypass_prefix",
            "bypass_suffix",
            "zero_input_seam",
            "zero_output_seam",
        }:
            raise ValueError(f"Unknown graft ablation {ablation!r}")
        if self.training and ablation != "none":
            raise ValueError(
                "Graft ablations are evaluation-only; training must execute every block"
            )
        hidden = self.embedding(input_ids)
        if ablation != "bypass_prefix":
            hidden = self.prefix(hidden, attention_mask)
        hidden = self.input_seam(hidden)
        if ablation == "zero_input_seam":
            hidden = torch.zeros_like(hidden)
        donor_input = hidden
        checkpointing = self.training and self.gradient_checkpointing
        if ablation != "bypass_graft":
            hidden = self.graft(hidden, attention_mask, gradient_checkpointing=checkpointing)
        if ablation == "zero_graft":
            hidden = torch.zeros_like(hidden)
        if self.donor_delta and ablation == "none":
            # Only the donor's transformation reaches the suffix. Under the
            # bypass/zero ablations the un-subtracted signal is retained so
            # donor-removed and donor-neutralized remain distinct tests.
            hidden = hidden - donor_input
        hidden = self.output_seam(hidden)
        if ablation == "zero_output_seam":
            hidden = torch.zeros_like(hidden)
        if ablation != "bypass_suffix":
            hidden = self.suffix(hidden, attention_mask, gradient_checkpointing=checkpointing)
        hidden = self.final_norm(hidden).to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden).float()
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long:
                raise ValueError("labels must be int64 and match input_ids")
            if torch.any((attention_mask == 0) & (labels != -100)):
                raise ValueError("Padding must have label -100")
            valid = labels != -100
            if torch.any(valid & ((labels < 0) | (labels >= logits.shape[-1]))):
                raise ValueError("Labels must be recipient token IDs or -100")
            if not torch.any(labels[:, 1:] != -100):
                raise ValueError("There must be at least one supervised next-token target")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return SurgeryOutput(logits, loss)

    def structure_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "recipient_family": self.recipient_config.model_type,
            "donor_family": self.donor_config.model_type,
            "recipient_width": self.recipient_config.hidden_size,
            "donor_width": self.donor_config.hidden_size,
            "prefix_blocks": len(self.prefix.layers),
            "transplanted_blocks": len(self.graft.layers),
            "suffix_blocks": len(self.suffix.layers),
            "removed_recipient_blocks": self.recipe.recipient_cut.stop
            - self.recipe.recipient_cut.start,
            "unique_parameters": total,
            "trainable_dense_seam_parameters": trainable,
            "frozen_parameters": total - trainable,
            "context_limit": self.context_limit,
            "production_bypass": False,
            "parent_weight_arithmetic": False,
        }


def child_from_parents(recipe: Recipe, recipient: nn.Module, donor: nn.Module) -> SurgicalChild:
    recipe.validate(recipient.config, donor.config)
    a, b = extract_parent(recipient), extract_parent(donor)
    reference = a.embedding.weight
    if any(p.device != reference.device for p in donor.parameters()):
        raise ValueError("Put both parents on the same device before surgery")
    if any(p.dtype != reference.dtype for p in [*recipient.parameters(), *donor.parameters()]):
        raise ValueError(
            "Use one common floating-point dtype; mixed/quantized parents are unsupported"
        )
    cut, sector = recipe.recipient_cut, recipe.donor_sector
    ac = config_from_dict(a.config.to_dict())
    bc = config_from_dict(b.config.to_dict())
    return SurgicalChild(
        recipe,
        ac,
        bc,
        a.embedding,
        StackSegment(ac, a.layers[: cut.start]),
        StackSegment(bc, b.layers[sector.start : sector.stop]),
        StackSegment(ac, a.layers[cut.stop :]),
        a.norm,
        a.head,
    )


def empty_child(recipe: Recipe, recipient_config: dict, donor_config: dict) -> SurgicalChild:
    """Recreate ONLY retained modules; a checkpoint never requires either parent snapshot."""
    a, b = config_from_dict(recipient_config), config_from_dict(donor_config)
    recipe.validate(a, b)
    cut, sector = recipe.recipient_cut, recipe.donor_sector
    embedding = nn.Embedding(a.vocab_size, a.hidden_size, padding_idx=a.pad_token_id)
    head = nn.Linear(a.hidden_size, a.vocab_size, bias=False)
    if a.tie_word_embeddings:
        head.weight = embedding.weight
    return SurgicalChild(
        recipe,
        a,
        b,
        embedding,
        StackSegment(a, [make_block(a, i) for i in range(cut.start)]),
        StackSegment(b, [make_block(b, i) for i in range(sector.start, sector.stop)]),
        StackSegment(a, [make_block(a, i) for i in range(cut.stop, a.num_hidden_layers)]),
        make_norm(a),
        head,
    )
