"""Unpaired aggregate representation diagnostics. NEVER hidden-vector supervision targets."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch


class StreamMoments:
    """CPU float64 sufficient statistics; do not retain token/example activation vectors."""

    def __init__(self, width: int, *, covariance: bool = True):
        self.width = width
        self.covariance = covariance
        self.count = 0
        self.nonfinite = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.squares = torch.zeros(width, dtype=torch.float64)
        self.cross = torch.zeros(width, width, dtype=torch.float64) if covariance else None
        self.nearzero = 0
        self.large = 0
        self.maxabs = 0.0
        self.calls = 0
        self.positions = {}

    def update(self, hidden):
        x = hidden.detach().float().cpu()
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("Diagnostics expect unpadded [batch,sequence,width] activations")
        self.calls += 1
        self.nonfinite += int((~torch.isfinite(x)).sum())
        if self.nonfinite:
            raise FloatingPointError("Nonfinite native/bridge representation")
        flat = x.reshape(-1, self.width).double()
        self.count += len(flat)
        self.sum += flat.sum(0)
        self.squares += flat.square().sum(0)
        if self.cross is not None:
            self.cross += flat.T @ flat
        self.nearzero += int((x.abs() < 1e-6).sum())
        self.large += int((x.abs() > 10).sum())
        self.maxabs = max(self.maxabs, float(x.abs().max()))
        for index in range(x.shape[1]):
            bucket = min(9, index * 10 // x.shape[1])
            entry = self.positions.setdefault(bucket, {"elements": 0, "squares": 0.0})
            value = x[:, index]
            entry["elements"] += value.numel()
            entry["squares"] += float(value.square().sum())

    def covariance_matrix(self):
        if not self.count or self.cross is None:
            raise ValueError("Covariance not observed")
        mean = self.sum / self.count
        covariance = self.cross / self.count - torch.outer(mean, mean)
        return (covariance + covariance.T) / 2

    def summary(self):
        if not self.count:
            raise ValueError("Missing representation observations")
        mean = self.sum / self.count
        variance = (self.squares / self.count - mean.square()).clamp_min(0)
        result = {
            "kind": "measured_aggregate_not_teacher_target",
            "calls": self.calls,
            "vectors": self.count,
            "width": self.width,
            "channel_mean": mean.tolist(),
            "channel_variance": variance.tolist(),
            "global_mean": float(mean.mean()),
            "global_rms": float((self.squares.sum() / (self.count * self.width)).sqrt()),
            "nearzero_fraction": self.nearzero / (self.count * self.width),
            "large_abs_gt10_fraction": self.large / (self.count * self.width),
            "max_abs": self.maxabs,
            "nonfinite": self.nonfinite,
            "position_rms": [
                {"relative_bin": i, "rms": math.sqrt(v["squares"] / v["elements"])}
                for i, v in sorted(self.positions.items())
            ],
            "thresholds": {
                "nearzero_abs": 1e-6,
                "unusually_large_abs": 10,
                "saturation": "not defined universally for SiLU/linear; no fabricated rate",
            },
        }
        if self.cross is not None:
            covariance = self.covariance_matrix()
            eigen = torch.linalg.eigvalsh(covariance)
            denom = variance.sqrt().clamp_min(1e-8)
            corr = covariance / (denom[:, None] * denom[None, :])
            off = corr - torch.diag(corr.diag())
            result["covariance"] = {
                "trace": float(covariance.trace()),
                "eigen_min": float(eigen.min()),
                "eigen_max": float(eigen.max()),
                "offdiagonal_correlation_rms": float(off.square().mean().sqrt()),
                "effective_rank_energy99": int(
                    (eigen.flip(0).clamp_min(0).cumsum(0) < 0.99 * eigen.clamp_min(0).sum()).sum()
                    + 1
                ),
            }
        return result


@contextmanager
def capture_streams(specifications):
    """Map name -> (module, input-or-output, width); only aggregate buffers survive."""
    meters = {name: StreamMoments(width) for name, (_, _, width) in specifications.items()}
    hooks = []
    for name, (module, where, _) in specifications.items():

        def collect(_module, args, output, name=name, where=where):
            value = args[0] if where == "input" else output
            if isinstance(value, tuple):
                value = value[0]
            meters[name].update(value)

        hooks.append(module.register_forward_hook(collect))
    try:
        yield meters
    finally:
        for hook in hooks:
            hook.remove()


def gradient_summary(child):
    """Only new interface gradients; originals must never have gradients or trainable flags."""
    result = {}
    for name, p in child.named_parameters():
        allowed = name.startswith(("input_seam.", "output_seam."))
        if not allowed and (p.requires_grad or p.grad is not None):
            raise ValueError("A pretrained tensor entered the gradient/optimizer scope")
        if allowed:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError("Missing/nonfinite interface gradient")
            g = p.grad.detach().float()
            result[name] = {
                "l2_norm": float(g.norm()),
                "mean": float(g.mean()),
                "std": float(g.std(unbiased=False)),
                "max_abs": float(g.abs().max()),
                "zero_fraction": float((g == 0).float().mean()),
                "elements": g.numel(),
            }
    return result
