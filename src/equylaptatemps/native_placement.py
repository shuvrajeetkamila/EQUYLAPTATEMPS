"""Native-depth placement engine (Campaign 17).

Assembles a SurgicalChild whose executed blocks are an explicit
permutation of the host's own blocks. Deletion and duplication are
unrepresentable: any execution order that is not an exact
permutation of ``range(depth)`` is rejected before assembly. Every
deviation from the identity order must be declared in the returned
structural manifest. No tensor is reshaped, padded, renamed, or
reinitialized; blocks are the parents' own modules.
"""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn

from .alignment import install_interfaces
from .audit import module_fingerprint
from .backbones import StackSegment, extract_parent
from .model import DenseSeam, SurgicalChild


def validate_execution_order(execution_order: list[int], depth: int) -> None:
    if len(execution_order) != depth or sorted(execution_order) != list(range(depth)):
        raise ValueError(
            "Execution order must be an exact permutation of the host's blocks "
            f"(depth {depth}); deletion and duplication are not representable"
        )


def _shape_signature(block: nn.Module) -> tuple:
    return tuple((name, tuple(p.shape)) for name, p in block.named_parameters())


def build_native_depth_child(
    recipient_parent,
    *,
    execution_order: list[int],
    graft_span: tuple[int, int],
    seam_mode: str,
    recipe,
    donor_parent=None,
):
    """Build a depth-preserving child plus its structural manifest.

    execution_order: permutation of range(depth), in execution order.
    graft_span: (start, stop) in execution coordinates naming the
    declared replacement region (the graft segment). Blocks inside
    the graft span come from donor_parent when supplied (the
    candidate component), else from the recipient itself.
    seam_mode: "bypass" (nn.Identity) or "residual" (the established
    residual-identity interface install).
    """
    recipient = extract_parent(recipient_parent)
    depth = len(recipient.layers)
    validate_execution_order(list(execution_order), depth)
    start, stop = graft_span
    if not (0 <= start < stop <= depth):
        raise ValueError("graft_span must be a nonempty span within the execution order")
    donor = extract_parent(donor_parent) if donor_parent is not None else None
    if donor is not None and len(donor.layers) != depth:
        raise ValueError("Candidate component depth does not match the host depth")

    reference_signature = _shape_signature(recipient.layers[execution_order[0]])

    def take(source_index: int, *, from_donor: bool) -> nn.Module:
        block = (donor if from_donor else recipient).layers[source_index]
        if _shape_signature(block) != reference_signature:
            raise ValueError(
                f"Block {source_index} shape signature is incompatible with the host sector"
            )
        return block

    segments = {}
    manifest_blocks = []
    for label, lo, hi in (
        ("prefix", 0, start),
        ("graft", start, stop),
        ("suffix", stop, depth),
    ):
        modules = []
        for position in range(lo, hi):
            source_index = execution_order[position]
            from_donor = donor is not None and label == "graft"
            block = take(source_index, from_donor=from_donor)
            modules.append(block)
            manifest_blocks.append(
                {
                    "identity": f"block:{source_index}",
                    "original_index": source_index,
                    "final_index": position,
                    "segment": label,
                    "source_role": "graft_donor" if from_donor else "recipient",
                    "fingerprint_sha256": module_fingerprint(block),
                    "parameter_count": sum(p.numel() for p in block.parameters()),
                    "action": (
                        "replaced"
                        if label == "graft" and donor is not None
                        else "moved"
                        if source_index != position
                        else "untouched"
                    ),
                }
            )
        segments[label] = StackSegment(recipient.config, modules)

    child = SurgicalChild(
        recipe,
        recipient.config,
        (donor or recipient).config,
        recipient.embedding,
        segments["prefix"],
        segments["graft"],
        segments["suffix"],
        recipient.norm,
        recipient.head,
    )
    if seam_mode == "bypass":
        child.input_seam = nn.Identity()
        child.output_seam = nn.Identity()
        child.interfaces_config = {"architecture": "bypassed_for_diagnostics"}
    elif seam_mode == "residual":
        install_interfaces(child, {"architecture": "residual_identity"})
    else:
        raise ValueError(f"Unknown seam_mode {seam_mode!r}")

    manifest = {
        "depth_before": depth,
        "depth_after": sum(len(s.layers) for s in segments.values()),
        "execution_order": list(execution_order),
        "graft_span_execution_coordinates": [start, stop],
        "seam_mode": seam_mode,
        "blocks": manifest_blocks,
        "blocks_removed": [],
        "blocks_duplicated": [],
        "blocks_moved": [
            b["identity"] for b in manifest_blocks if b["original_index"] != b["final_index"]
        ],
        "total_block_parameters": sum(b["parameter_count"] for b in manifest_blocks),
    }
    return child, manifest


# ---------------------------------------------------------------------------
# Campaign 19: foreign-graft construction (preparation; not yet authorized
# for experimental use). Replaces exactly one declared recipient span with
# an equal-length foreign sector at the same depth. Recipient order outside
# the span is untouched; replaced recipient blocks are recorded, never
# silently dropped. Connectors are declared trainable DenseSeam bridges.
# ---------------------------------------------------------------------------


def build_foreign_graft_child(
    recipient_parent,
    donor_parent,
    *,
    recipient_span: tuple[int, int] = (11, 13),
    donor_span: tuple[int, int] = (11, 13),
    recipe,
    graft_blocks=None,
):
    """Replace recipient_span with an equal-length donor sector.

    graft_blocks optionally supplies pre-transformed donor blocks
    (scramble/random controls); the donor parent is never mutated.
    """

    recipient = extract_parent(recipient_parent)
    donor = extract_parent(donor_parent)
    depth = len(recipient.layers)
    r0, r1 = recipient_span
    d0, d1 = donor_span
    if not (0 <= r0 < r1 <= depth):
        raise ValueError("recipient_span outside recipient depth")
    if (r1 - r0) != (d1 - d0):
        raise ValueError(
            "Foreign sector length must equal the replaced recipient span "
            "length; native depth preservation forbids length-changing grafts"
        )
    if graft_blocks is None:
        graft_blocks = list(donor.layers[d0:d1])
    if len(graft_blocks) != r1 - r0:
        raise ValueError("Supplied graft blocks do not match the declared spans")
    segments = {
        "prefix": StackSegment(recipient.config, list(recipient.layers[:r0])),
        "graft": StackSegment(donor.config, list(graft_blocks)),
        "suffix": StackSegment(recipient.config, list(recipient.layers[r1:])),
    }
    child = SurgicalChild(
        recipe,
        recipient.config,
        donor.config,
        recipient.embedding,
        segments["prefix"],
        segments["graft"],
        segments["suffix"],
        recipient.norm,
        recipient.head,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe.seed)
        child.input_seam = DenseSeam(recipient.config.hidden_size, donor.config.hidden_size)
        child.output_seam = DenseSeam(donor.config.hidden_size, recipient.config.hidden_size)
    child.interfaces_config = {
        "architecture": "dense_bridge",
        "initialization": "seeded_default",
        "seed": recipe.seed,
    }
    blocks = []
    for i in range(depth):
        if r0 <= i < r1:
            blocks.append(
                {
                    "identity": f"recipient:block:{i}",
                    "original_index": i,
                    "final_index": None,
                    "source_role": "recipient",
                    "action": "replaced",
                }
            )
        else:
            blocks.append(
                {
                    "identity": f"recipient:block:{i}",
                    "original_index": i,
                    "final_index": i if i < r0 else i,
                    "source_role": "recipient",
                    "fingerprint_sha256": module_fingerprint(recipient.layers[i]),
                    "action": "untouched",
                }
            )
    for j, block in enumerate(graft_blocks):
        blocks.append(
            {
                "identity": f"donor:block:{d0 + j}",
                "original_index": d0 + j,
                "final_index": r0 + j,
                "source_role": "foreign_donor",
                "fingerprint_sha256": module_fingerprint(block),
                "action": "inserted",
            }
        )
    manifest = {
        "depth_before": depth,
        "depth_after": depth,
        "recipient_span_replaced": [r0, r1],
        "donor_span_inserted": [d0, d1],
        "recipient_width": recipient.config.hidden_size,
        "donor_width": donor.config.hidden_size,
        "blocks": blocks,
        "blocks_removed_unrelated": [],
        "blocks_duplicated": [],
        "trainable_components": ["input_seam", "output_seam"],
    }
    return child, manifest


def _digest(param) -> str:
    return hashlib.sha256(param.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def scramble_blocks(blocks, seed: int):
    """Campaign 11 within-tensor permutation on deep-copied blocks.

    Preserves every eligible (2D) tensor's exact value multiset while
    destroying learned arrangement; 1D tensors are excluded and
    recorded. The source blocks are never mutated.
    """
    copied = copy.deepcopy(list(blocks))
    generator = torch.Generator().manual_seed(seed)
    record = []
    for index, block in enumerate(copied):
        for name, param in block.named_parameters():
            eligible = param.ndim == 2
            before = _digest(param)
            preserved = True
            if eligible:
                with torch.no_grad():
                    flat = param.detach().reshape(-1).clone()
                    permuted = flat[torch.randperm(flat.numel(), generator=generator)]
                    param.copy_(permuted.reshape(param.shape))
                    preserved = bool(
                        torch.equal(torch.sort(flat).values, torch.sort(permuted).values)
                    )
            record.append(
                {
                    "block": index,
                    "name": name,
                    "eligible": eligible,
                    "multiset_preserved": preserved,
                    "sha256_before": before,
                    "sha256_after": _digest(param),
                }
            )
    for block in copied:
        block.requires_grad_(False)
    return copied, {
        "kind": "within_tensor_permutation",
        "seed": seed,
        "tensors": record,
        "all_multisets_preserved": all(r["multiset_preserved"] for r in record),
    }


def randomize_blocks(blocks, seed: int):
    """Distribution-matched random control on deep-copied blocks.

    Every eligible (2D) tensor is redrawn from a normal distribution
    with the original tensor's mean and standard deviation (seeded);
    shapes and parameter counts are identical, learned arrangement
    and cross-tensor structure are absent.
    """
    copied = copy.deepcopy(list(blocks))
    generator = torch.Generator().manual_seed(seed)
    record = []
    for index, block in enumerate(copied):
        for name, param in block.named_parameters():
            eligible = param.ndim == 2
            before = _digest(param)
            if eligible:
                with torch.no_grad():
                    mean = float(param.float().mean())
                    std = float(param.float().std())
                    param.copy_(
                        torch.randn(param.shape, generator=generator, dtype=torch.float32)
                        .mul(std)
                        .add(mean)
                        .to(param.dtype)
                    )
            record.append(
                {
                    "block": index,
                    "name": name,
                    "eligible": eligible,
                    "sha256_before": before,
                    "sha256_after": _digest(param),
                }
            )
    for block in copied:
        block.requires_grad_(False)
    return copied, {"kind": "distribution_matched_random", "seed": seed, "tensors": record}
