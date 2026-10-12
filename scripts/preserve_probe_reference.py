#!/usr/bin/env python3
"""Preserve an externally measured probe, checking that no frozen parent tensors are included."""

import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path("artifacts/measured-pretrained")
    report = json.loads((root / "results.json").read_text())
    if (
        report["status"] != "PASS"
        or report["source_commit"] != "f812e6b5d74e7465e02cd6f41df71589dc35ad60"
    ):
        raise ValueError("Not the independently observed successful structural probe")
    reference = root / "reference-child"
    descriptor = json.loads((reference / "reference.json").read_text())
    weights = reference / "interfaces.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != descriptor["interfaces_sha256"] or descriptor["parent_kind"] != "pretrained":
        raise ValueError("Actual reference checkpoint digest/kind mismatch")
    if descriptor["frozen_sha256"] != report["frozen_invariance"]["frozen_sha256"]:
        raise ValueError(
            "Reference does not identify the measured pretrained child's frozen tensors"
        )
    with weights.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if not 0 < length < 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(handle.read(length))
    expected = {
        f"{side}_seam.{layer}.{part}"
        for side in ("input", "output")
        for layer in ("norm", "projection")
        for part in ("weight", "bias")
    }
    if set(header) - {"__metadata__"} != expected or weights.stat().st_size > 10 * 1024**2:
        raise ValueError("Reference contains unexpected/large tensor payloads")
    if descriptor["capability_status"] != "UNPROVEN":
        raise ValueError("Do not publish an unsupported inheritance/capability claim")
    target = Path("examples/pretrained-structural-reference")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(weights, target / weights.name)
    shutil.copy2(reference / "reference.json", target / "reference.json")
    shutil.copy2(root / "calibration-report.json", target / "calibration-report.json")
    shutil.copy2(root / "sector-inventory.json", target / "sector-inventory.json")
    (target / "interfaces.safetensors.sha256").write_text(digest + "  interfaces.safetensors\n")
    (target / "README.md").write_text(
        "# Actual pretrained structural child: bridge-only checkpoint\n\n"
        "HETEROGENEOUS AI MODEL SURGERY. Measured on GitHub run 38290378364, source f812e6b.\n\n"
        "This contains ONLY the eight newly trained full-sized interface/norm tensors. "
        "The real Mamba and Qwen frozen tensors are NOT bundled. This is NOT standalone.\n\n"
        "217,553,920 integrated parameters: 215,187,712 frozen and 2,366,208 new trainable. "
        "Whole Qwen blocks [13,15) replace Mamba blocks [11,13). "
        "Eight steps on four short WikiText TRAIN windows; no distillation.\n\n"
        "Forward, checkpoint export, and exact fresh-process reload were measured. "
        "A generation sample was `io`; no useful capability or inheritance established. "
        "Do not treat fitting loss as a benchmark score.\n\n"
        "Reconstruct on a connected >=12 GiB RAM machine with the repository installed:\n\n"
        "```bash\npython scripts/reconstruct_reference.py --reference "
        "examples/pretrained-structural-reference --device cpu\n```\n\n"
        "Requires ~3.6 GB pinned public parent safetensors, common float32 loading, and "
        "the exact recipe/provenance in reference.json. Frozen/graft and tokenizer hashes "
        "are verified. Use --out for a large private standalone child.\n\n"
        "Code is Apache-2.0. Dataset terms remain: WikiText CC-BY-SA-3.0/GFDL; "
        "see the pinned source/card in the dated research report. Qwen is Apache-2.0. "
        "Mamba's official original checkpoint declares Apache-2.0, but the HF conversion "
        "omits a direct license tag/document. No frozen weights are redistributed here.\n"
    )
    print("Preserved actual new interfaces:", weights.stat().st_size, "bytes; SHA-256", digest)


if __name__ == "__main__":
    main()
