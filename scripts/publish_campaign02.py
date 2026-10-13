#!/usr/bin/env python3
"""Preserve campaign metadata after a bounded/interrupted job; no claimed completed result."""

import argparse
from pathlib import Path

from run_pretrained_study import publish_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--post-pr", type=int, required=True)
    args = p.parse_args()
    out = Path(args.out)
    if not out.is_dir():
        p.error("No initialized campaign directory exists")
    publish_bundle(out, args.post_pr)


if __name__ == "__main__":
    main()
