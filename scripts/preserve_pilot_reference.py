#!/usr/bin/env python3
"""Preserve actual selected negative-result pilot interfaces, never frozen model weights."""

import hashlib
import json
import shutil
from pathlib import Path


def main():
    source = Path("artifacts/measured-pilot")
    report = json.loads((source / "results.json").read_text())
    if report["source_commit"] != "95b2831a7f54ae7755e6892e364d993a779a3193":
        raise ValueError("Unexpected measured-pilot source commit")
    if any(stage["status"] != "PASS" for stage in report["stages"].values()):
        raise ValueError("Do not present incomplete experimental execution as complete")
    reference = source / "reference-child"
    descriptor = json.loads((reference / "reference.json").read_text())
    weights = reference / "interfaces.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != descriptor["interfaces_sha256"] or descriptor["parent_kind"] != "pretrained":
        raise ValueError("Actual pilot reference checksum/kind mismatch")
    if descriptor["frozen_sha256"] != report["frozen_invariance"]["child_frozen_sha256"]:
        raise ValueError("Reference differs from the measured selected child")
    expected = {
        f"{side}_seam.{layer}.{part}"
        for side in ("input", "output")
        for layer in ("norm", "projection")
        for part in ("weight", "bias")
    }
    with weights.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if not 0 < length < 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        header = json.loads(handle.read(length))
    if set(header) - {"__metadata__"} != expected or weights.stat().st_size > 10 * 1024**2:
        raise ValueError("Payload must contain new interfaces only")
    target = Path("examples/pretrained-pilot-reference")
    target.mkdir(parents=True, exist_ok=False)
    for file in ("reference.json", "interfaces.safetensors"):
        shutil.copy2(reference / file, target / file)
    (target / "interfaces.safetensors.sha256").write_text(digest + "  interfaces.safetensors\n")
    (target / "README.md").write_text(
        "# Actual selected pretrained pilot child — NEGATIVE capability result\n\n"
        "HETEROGENEOUS AI MODEL SURGERY. Connected run 38293921543, source 95b2831.\n\n"
        "New interfaces ONLY, not standalone; frozen parents are not distributed. "
        "Whole Qwen blocks [20,22) replace Mamba [11,13), selected on development NLL. "
        "Sixteen interface steps per candidate on actual human GSM8K TRAIN answers "
        "and real WikiText TRAIN text. No distillation or parent-weight changes.\n\n"
        "Held-out pilot: child math 0/4, WikiText 3.444856383352949 bits/byte, "
        "worse than both original parents (Mamba 1.765828132006639; Qwen 2.5806685715288404). "
        "Child generations repeated `the` and commas. No useful inheritance established. "
        "Formal capability criteria remain NOT_RUN because the sample minimum is 64.\n\n"
        "```bash\npython scripts/reconstruct_reference.py --reference "
        "examples/pretrained-pilot-reference --device cpu\n```\n\n"
        "Requires exact pinned parent safetensors and common float32; identities, "
        "tokenizers and frozen/graft hashes verified. See the dated report/raw observations "
        "and docs/pretrained-study.md. Parent and dataset terms/gaps remain; "
        "no frozen tensor redistribution and no public standalone model release claimed.\n"
    )
    print("Preserved negative-result child interfaces", weights.stat().st_size, digest)


if __name__ == "__main__":
    main()
