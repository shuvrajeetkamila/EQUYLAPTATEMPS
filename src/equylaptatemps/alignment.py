"""Small full-sized bridge family and unpaired statistical initialization, NOT distillation."""

from __future__ import annotations

import torch
from torch import nn

from .model import DenseSeam


class RMSCondition(nn.Module):
    def __init__(self, width, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon).to(x.dtype)
            * self.weight
        )


def install_interfaces(child, design):
    """Only pointwise interface operations change. Every native parent block stays intact."""
    allowed = {"layernorm_linear", "linear", "rmsnorm_linear"}
    if design["architecture"] not in allowed:
        raise ValueError("Unregistered interface architecture")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(child.recipe.seed)
        seams = [
            DenseSeam(child.recipient_config.hidden_size, child.donor_config.hidden_size),
            DenseSeam(child.donor_config.hidden_size, child.recipient_config.hidden_size),
        ]
    for seam in seams:
        width = seam.projection.in_features
        if design["architecture"] == "linear":
            seam.norm = nn.Identity()
        elif design["architecture"] == "rmsnorm_linear":
            seam.norm = RMSCondition(width)
        seam.to(device=child.device, dtype=child.embedding.weight.dtype)
    child.input_seam, child.output_seam = seams
    child.interfaces_config = dict(design)
    child.assert_trainable_contract()
    child.eval()
    return child


def whitening_matrix(covariance, *, shrinkage=0.1, floor=0.01):
    """Source-only covariance conditioning; no donor activation vectors are targets."""
    covariance = covariance.double()
    scale = float(covariance.diag().mean())
    if scale <= 0 or not torch.isfinite(covariance).all():
        raise ValueError("Invalid observed covariance")
    regularized = (1 - shrinkage) * covariance + shrinkage * scale * torch.eye(
        len(covariance), dtype=torch.float64
    )
    eigen, basis = torch.linalg.eigh((regularized + regularized.T) / 2)
    inverse = (basis * eigen.clamp_min(floor * scale).rsqrt()) @ basis.T
    return inverse.float()


def coloring_matrix(covariance, *, floor=0.01):
    """Symmetric square root of a covariance: the linear map that *gives* a
    whitened stream this covariance (Campaign 9 donor-native geometry)."""
    covariance = covariance.double()
    scale = float(covariance.diag().mean())
    if scale <= 0 or not torch.isfinite(covariance).all():
        raise ValueError("Invalid observed covariance")
    eigen, basis = torch.linalg.eigh((covariance + covariance.T) / 2)
    root = (basis * eigen.clamp_min(floor * scale).sqrt()) @ basis.T
    return root.float()


def orthogonal_pair(child, *, seed):
    """A fixed coordinate-conditioning control, not identity across unequal token semantics."""
    r = child.recipient_config.hidden_size
    d = child.donor_config.hidden_size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if d >= r:
            q = torch.linalg.qr(torch.randn(d, r), mode="reduced").Q
        else:
            q = torch.linalg.qr(torch.randn(r, d), mode="reduced").Q.T
    with torch.no_grad():
        child.input_seam.projection.weight.copy_(q)
        child.output_seam.projection.weight.copy_(q.T)
        child.input_seam.projection.bias.zero_()
        child.output_seam.projection.bias.zero_()


def initialize_conditioned(child, method, *, statistics=None, seed=42):
    """Stats are aggregate calibration moments, never paired/per-token imitation labels.

    Residual RMS scales are initialization constants estimated independently on
    unlabeled source TRAIN text. The subsequent loss is the CHILD's own text loss.
    """
    if method == "random":
        return {"kind": "random_same_seed_full_dense", "teacher_targets": False}
    allowed = {
        "fixed_orthogonal",
        "residual_scale",
        "covariance_orthogonal",
        "covariance_orthogonal_noscale",
        "donor_native_geometry",
        "natgeom_in_trained_out",
    }
    if method not in allowed:
        raise ValueError("Unknown alignment initialization")
    orthogonal_methods = {
        "fixed_orthogonal",
        "covariance_orthogonal",
        "covariance_orthogonal_noscale",
        "donor_native_geometry",
        "natgeom_in_trained_out",
    }
    saved_output = None
    if method == "natgeom_in_trained_out":
        # Campaign 10: the output pathway must start from the generic
        # constructed DenseSeam default (fully trained, no geometry
        # preconditioning). Preserve it before the orthogonal pair rewrites
        # both seams; it is restored after the input seam is geometrized.
        saved_output = {k: v.detach().clone() for k, v in child.output_seam.state_dict().items()}
    if method in orthogonal_methods:
        orthogonal_pair(child, seed=seed)
    record = {
        "kind": "unpaired_distribution_initialization",
        "method": method,
        "teacher_targets": False,
    }
    if method == "fixed_orthogonal":
        record["limitation"] = (
            "Not an identity mapping of token or hidden semantics; native graft still mandatory"
        )
        return record
    if statistics is None:
        raise ValueError("Distribution-aware init requires real TRAIN measurements")
    streams = statistics["summary"]["streams"]
    with torch.no_grad():
        if method in {"covariance_orthogonal", "covariance_orthogonal_noscale"}:
            for role, key in [
                ("input_seam", "child_input_normalized"),
                ("output_seam", "child_output_normalized"),
            ]:
                inverse = whitening_matrix(statistics["tensors"][key + "/covariance"])
                seam = getattr(child, role)
                seam.projection.weight.copy_(seam.projection.weight @ inverse)
        if method == "natgeom_in_trained_out":
            # Campaign 10: input seam gets Campaign 9's donor-native geometry
            # exactly; the output seam is restored below to its generic
            # trained initialization.
            inverse = whitening_matrix(statistics["tensors"]["child_input_normalized/covariance"])
            color = coloring_matrix(statistics["tensors"]["donor_native_input/covariance"])
            child.input_seam.projection.weight.copy_(
                color @ child.input_seam.projection.weight @ inverse
            )
        if method == "donor_native_geometry":
            # Whiten the child-side stream, then color it with the *target*
            # side's native covariance: donor-native geometry going in,
            # recipient-native geometry coming out. Identical donor-native
            # matrices for pretrained and twin children (architecture-level
            # statistics); each child's whitening uses its own measured
            # stream, so the sector weights are the only difference.
            for role, child_key, native_key in [
                ("input_seam", "child_input_normalized", "donor_native_input"),
                ("output_seam", "child_output_normalized", "recipient_native_output"),
            ]:
                inverse = whitening_matrix(statistics["tensors"][child_key + "/covariance"])
                color = coloring_matrix(statistics["tensors"][native_key + "/covariance"])
                seam = getattr(child, role)
                seam.projection.weight.copy_(color @ seam.projection.weight @ inverse)
        # Scalars only. No hidden target vector, no parent-response or hidden imitation loss.
        in_scale = streams["donor_native_input"]["global_rms"] / max(
            streams["child_donor_input"]["global_rms"], 1e-6
        )
        out_scale = streams["recipient_native_output"]["global_rms"] / max(
            streams["child_host_suffix_input"]["global_rms"], 1e-6
        )
        if method == "natgeom_in_trained_out":
            child.output_seam.load_state_dict(saved_output)
            applied_in = in_scale if 0.05 < in_scale < 100 else None
            if applied_in is not None:
                child.input_seam.projection.weight.mul_(applied_in)
            record.update(
                input_gain=in_scale,
                output_gain=None,
                applied_gains={"input_seam": applied_in, "output_seam": None},
                output_pathway="trained_default_init_no_geometry",
                limitation=(
                    "Input geometry only; output pathway is generically initialized "
                    "and trained. Moment conditioning is NOT semantic alignment evidence"
                ),
            )
            return record
        if method in {"covariance_orthogonal_noscale", "donor_native_geometry"}:
            # Campaign 8 Amendment 01: a donor whose native residual stream is
            # far outside the Campaign-2-era safe gain range (e.g. SmolLM late
            # blocks, ~186x) would otherwise make calibration impossible.
            # Per-token RMSNorm at every donor block entry cancels pure input
            # scale, so an out-of-range gain is skipped and recorded instead of
            # raising; whitening and the orthogonal pair still apply.
            applied = {}
            for role, gain in (("input_seam", in_scale), ("output_seam", out_scale)):
                if 0.05 < gain < 100:
                    getattr(child, role).projection.weight.mul_(gain)
                    applied[role] = gain
                else:
                    applied[role] = None
            record.update(
                input_gain=in_scale,
                output_gain=out_scale,
                applied_gains=applied,
                limitation=(
                    "Out-of-range RMS gains skipped (RMSNorm cancels input scale); "
                    "moment conditioning is NOT semantic alignment evidence"
                ),
            )
            return record
        if not 0.05 < in_scale < 100 or not 0.05 < out_scale < 100:
            raise ValueError("Unsafe observed scale calibration")
        child.input_seam.projection.weight.mul_(in_scale)
        child.output_seam.projection.weight.mul_(out_scale)
        record.update(
            input_gain=in_scale,
            output_gain=out_scale,
            limitation="Marginal moment conditioning is NOT semantic alignment evidence",
        )
    return record
