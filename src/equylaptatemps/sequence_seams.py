"""Sequence-aware full-sized interfaces for Campaign 3 (NOT LoRA, NOT distillation).

Every mixer is a newly initialized, explicitly declared trainable component that
sits *after* the same full dense projection used by Campaign 2.  Each mixer is
residual and zero-initialized at its output, so at initialization a sequence
seam is numerically identical to the corresponding Campaign 2 dense seam.
Any later difference is therefore attributable to learned sequence mixing,
not to a changed starting point.

Causality contract: the representation at position t may depend only on
positions <= t.  The convolutional mixer uses left padding only; the
attention mixer uses an explicit causal mask; the recurrent mixer consumes
the sequence left to right.  The child is evaluated with right padding, so
real positions never attend to later (padded) positions.

Parent blocks remain frozen; only ``input_seam.*`` / ``output_seam.*``
parameters train, exactly as in Campaign 2.  No parent hidden activations,
parent output distributions, or teacher-generated targets are used anywhere.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SEQUENCE_KINDS = ("causal_conv", "causal_attention", "bottleneck_gru")


class CausalDepthwiseConvMixer(nn.Module):
    """Depthwise causal convolution delta with a kernel over past positions only."""

    def __init__(self, width: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("A sequence mixer needs a receptive field of at least 2")
        self.width = width
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(width, width, kernel_size, groups=width)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # [B, T, W] -> [B, W, T]; pad the past only, never the future.
        sequence = hidden.transpose(1, 2)
        padded = F.pad(sequence, (self.kernel_size - 1, 0))
        return self.conv(padded).transpose(1, 2)


class BottleneckCausalAttentionMixer(nn.Module):
    """Single causal self-attention block in a small bottleneck space.

    The bottleneck keeps this interface far smaller than the donor sector it
    serves (documented parameter counts are emitted by the experiment runner);
    it is a representation adapter, not a replacement language model.
    """

    def __init__(self, width: int, bottleneck: int = 128, heads: int = 4, max_positions: int = 512):
        super().__init__()
        if bottleneck % heads:
            raise ValueError("Bottleneck width must divide evenly across heads")
        self.width = width
        self.bottleneck = bottleneck
        self.heads = heads
        self.head_dim = bottleneck // heads
        self.down = nn.Linear(width, bottleneck)
        self.position = nn.Embedding(max_positions, bottleneck)
        nn.init.zeros_(self.position.weight)
        self.qkv = nn.Linear(bottleneck, 3 * bottleneck)
        self.out = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.norm = nn.LayerNorm(bottleneck)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        positions = self.position.weight[:length]
        if length > self.position.num_embeddings:
            positions = torch.cat(
                [
                    positions,
                    positions.new_zeros(length - self.position.num_embeddings, self.bottleneck),
                ]
            )
        state = self.norm(self.down(hidden) + positions)
        qkv = self.qkv(state).view(hidden.shape[0], length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.permute(0, 2, 1, 3).reshape(hidden.shape[0], length, self.bottleneck)
        return self.out(attended)


class BottleneckGRUMixer(nn.Module):
    """Left-to-right recurrent (state-space style) bottleneck mixer."""

    def __init__(self, width: int, bottleneck: int = 128):
        super().__init__()
        self.width = width
        self.bottleneck = bottleneck
        self.down = nn.Linear(width, bottleneck)
        self.gru = nn.GRU(bottleneck, bottleneck, batch_first=True)
        self.out = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        state, _ = self.gru(self.down(hidden))
        return self.out(state)


def make_mixer(kind: str, width: int, design: dict) -> nn.Module:
    if kind == "causal_conv":
        return CausalDepthwiseConvMixer(width, kernel_size=int(design.get("kernel_size", 5)))
    if kind == "causal_attention":
        return BottleneckCausalAttentionMixer(
            width,
            bottleneck=int(design.get("bottleneck", 128)),
            heads=int(design.get("heads", 4)),
            max_positions=int(design.get("max_positions", 512)),
        )
    if kind == "bottleneck_gru":
        return BottleneckGRUMixer(width, bottleneck=int(design.get("bottleneck", 128)))
    raise ValueError(f"Unknown sequence mixer {kind!r}")


class SequenceSeam(nn.Module):
    """LayerNorm -> full dense projection -> residual causal sequence mixer.

    Exposes ``norm`` and ``projection`` with the same shapes as Campaign 2's
    ``DenseSeam`` so the existing unpaired statistical initialization
    (covariance whitening + RMS gains) applies unchanged.  Hidden activations
    are never used as training targets; statistics only set initial scales.
    """

    def __init__(self, source_width: int, destination_width: int, design: dict):
        super().__init__()
        self.design = dict(design)
        self.norm = nn.LayerNorm(source_width)
        self.projection = nn.Linear(source_width, destination_width, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.mixer = make_mixer(design["kind"], destination_width, design)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.projection.weight.dtype)
        projected = self.projection(self.norm(hidden))
        return projected + self.mixer(projected)


def install_sequence_interfaces(child, design: dict):
    """Replace only the pointwise seams with sequence-aware seams.

    ``design`` example: ``{"family": "sequence", "kind": "causal_conv",
    "kernel_size": 5}``.  Every native parent block stays intact and frozen.
    """
    if design.get("family") != "sequence" or design.get("kind") not in SEQUENCE_KINDS:
        raise ValueError(f"Unregistered sequence interface design {design!r}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(child.recipe.seed)
        seams = [
            SequenceSeam(
                child.recipient_config.hidden_size, child.donor_config.hidden_size, design
            ),
            SequenceSeam(
                child.donor_config.hidden_size, child.recipient_config.hidden_size, design
            ),
        ]
    for seam in seams:
        seam.to(device=child.device, dtype=child.embedding.weight.dtype)
    child.input_seam, child.output_seam = seams
    child.interfaces_config = dict(design)
    child.donor_delta = bool(design.get("donor_delta", False))
    child.assert_trainable_contract()
    child.eval()
    return child


def sequence_design_report(child) -> dict:
    """Plain accounting of what is new, trainable, frozen, and causal."""
    trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
    total = sum(p.numel() for p in child.parameters())
    mixer = sum(
        p.numel()
        for seam in (child.input_seam, child.output_seam)
        for name, p in seam.named_parameters()
        if name.startswith("mixer.")
    )
    return {
        "interfaces_config": getattr(child, "interfaces_config", None),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "sequence_mixer_parameters": mixer,
        "trainable_fraction_of_child": trainable / total,
        "causality": "left-padded depthwise conv / causal-masked attention / left-to-right GRU",
        "parent_blocks_modified": False,
    }


class BottleneckSequenceSeam(nn.Module):
    """Norm -> down-projection -> causal mixer -> up-projection.

    The narrow bottleneck (width << both parent widths) caps how much
    information the two interfaces can pass between themselves, so the
    donor sector is the only high-capacity transformation on the path.
    It cannot by itself prove donor use; the random-twin and ablation
    gates remain mandatory.
    """

    def __init__(self, source_width: int, destination_width: int, design: dict):
        super().__init__()
        self.design = dict(design)
        bottleneck = int(design["bottleneck"])
        self.norm = nn.LayerNorm(source_width)
        self.down = nn.Linear(source_width, bottleneck, bias=True)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.mixer = make_mixer(design["kind"], bottleneck, design)
        self.up = nn.Linear(bottleneck, destination_width, bias=True)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.down.weight.dtype)
        state = self.down(self.norm(hidden))
        return self.up(state + self.mixer(state))


def install_bottleneck_interfaces(child, design: dict):
    """Install bottleneck seams; optionally enable the donor-delta pathway."""
    if design.get("family") != "bottleneck" or design.get("kind") not in SEQUENCE_KINDS:
        raise ValueError(f"Unregistered bottleneck interface design {design!r}")
    if (
        not 4
        <= int(design["bottleneck"])
        < min(child.recipient_config.hidden_size, child.donor_config.hidden_size)
    ):
        raise ValueError("Bottleneck must be narrower than both parent widths")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(child.recipe.seed)
        seams = [
            BottleneckSequenceSeam(
                child.recipient_config.hidden_size, child.donor_config.hidden_size, design
            ),
            BottleneckSequenceSeam(
                child.donor_config.hidden_size, child.recipient_config.hidden_size, design
            ),
        ]
    for seam in seams:
        seam.to(device=child.device, dtype=child.embedding.weight.dtype)
    child.input_seam, child.output_seam = seams
    child.interfaces_config = dict(design)
    child.donor_delta = bool(design.get("donor_delta", False))
    child.assert_trainable_contract()
    child.eval()
    return child


def initialize_bottleneck(child, statistics):
    """Covariance whitening on `down`, scalar RMS gains on `up`.

    Same unpaired aggregate statistics and formulas as Campaign 2/3's
    covariance initialization; no hidden activations are used as targets.
    """
    from .alignment import whitening_matrix

    streams = statistics["summary"]["streams"]
    with torch.no_grad():
        for role, key in [
            ("input_seam", "child_input_normalized"),
            ("output_seam", "child_output_normalized"),
        ]:
            seam = getattr(child, role)
            inverse = whitening_matrix(statistics["tensors"][key + "/covariance"])
            seam.down.weight.copy_(seam.down.weight @ inverse)
        in_scale = streams["donor_native_input"]["global_rms"] / max(
            streams["child_donor_input"]["global_rms"], 1e-6
        )
        out_scale = streams["recipient_native_output"]["global_rms"] / max(
            streams["child_host_suffix_input"]["global_rms"], 1e-6
        )
        if not 0.05 < in_scale < 100 or not 0.05 < out_scale < 100:
            raise ValueError("Unsafe observed scale calibration")
        child.input_seam.up.weight.mul_(in_scale)
        child.output_seam.up.weight.mul_(out_scale)
    return {
        "kind": "unpaired_distribution_initialization",
        "method": "bottleneck_covariance_orthogonal",
        "teacher_targets": False,
        "input_gain": in_scale,
        "output_gain": out_scale,
        "limitation": "Marginal moment conditioning is NOT semantic alignment evidence",
    }
