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

from torch import nn

from .alignment import install_interfaces
from .audit import module_fingerprint
from .backbones import StackSegment, extract_parent
from .model import SurgicalChild


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
