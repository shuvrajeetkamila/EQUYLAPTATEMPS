#!/usr/bin/env python3
"""Fresh-process child-only load: no parent loading/network is allowed by this check."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from equylaptatemps.artifacts import load_artifact
from equylaptatemps.audit import hash_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def forbid_parent_loading(*_args, **_kwargs):
        raise RuntimeError("Fresh-process check forbids any parent model download/loading")

    AutoModelForCausalLM.from_pretrained = forbid_parent_loading
    stored = load_artifact(args.child)
    ids = torch.tensor(json.loads(Path(args.ids).read_text()), dtype=torch.long)
    with torch.inference_mode():
        logits = stored.model(ids).logits
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"logits": logits}, str(out / "logits.safetensors"))
    result = {
        "status": "PASS",
        "finite_logits": bool(torch.isfinite(logits).all()),
        "shape": list(logits.shape),
        "logits_sha256": hash_state({"logits": logits}),
        "parents_loaded": False,
        "parent_kind": "synthetic" if stored.model.recipe.synthetic else "pretrained",
        "last_token_argmax": int(logits[0, -1].argmax()),
    }
    (out / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
