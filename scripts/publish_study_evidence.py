#!/usr/bin/env python3
"""Publish partial/failure evidence even when an explicitly bounded experiment is interrupted."""

import argparse
import json
from pathlib import Path

from run_pretrained_study import publish_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--post-pr", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not (out / "results.json").exists():
        parser.error("No initialized result file to publish")
    path = out / "results.json"
    result = json.loads(path.read_text())
    result["interrupted_or_failed_publication"] = True
    if (
        result.get("kind") == "real_pretrained_language_diagnostic"
        and result["status"] == "NOT_RUN"
    ):
        result["status"] = "FAIL"
        result["error"] = {
            "type": "Interrupted",
            "reason": "Bounded runner interrupted before completion",
        }
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    publish_bundle(out, args.post_pr)


if __name__ == "__main__":
    main()
