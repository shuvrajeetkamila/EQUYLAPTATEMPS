#!/usr/bin/env python3
"""Fresh-process standalone child decode, with original parent loading forbidden."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from equylaptatemps.artifacts import load_artifact
from equylaptatemps.decoding import cached_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    def forbid(*_args, **_kwargs):
        raise RuntimeError("Cached child-only process cannot load original parents")

    AutoModelForCausalLM.from_pretrained = forbid
    stored = load_artifact(args.child)
    inputs = json.loads(Path(args.inputs).read_text())
    result = cached_generate(
        stored.model,
        stored.tokenizer,
        inputs["prompt"],
        max_new_tokens=inputs["max_new_tokens"],
        max_input_tokens=2048,
    )
    identical = (
        result.text == inputs["text"]
        and result.generated_ids == inputs["generated_ids"]
        and result.stop_reason == inputs["stop_reason"]
    )
    report = {
        "status": "PASS" if identical else "FAIL",
        "generation_identical": identical,
        "parents_loaded": False,
        "parent_kind": "synthetic" if stored.model.recipe.synthetic else "pretrained",
        "text": result.text,
        "generated_ids": result.generated_ids,
        "stop_reason": result.stop_reason,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not identical:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
