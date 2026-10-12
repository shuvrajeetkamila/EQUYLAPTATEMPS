#!/usr/bin/env python3
"""Recompute and verify saved real-study evidence without loading pretrained weights."""

import argparse
import json

from equylaptatemps.evidence import validate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--datasets", help="Optional already-acquired registered dataset cache")
    args = parser.parse_args()
    print(json.dumps(validate_results(args.experiment, dataset_root=args.datasets), indent=2))


if __name__ == "__main__":
    main()
