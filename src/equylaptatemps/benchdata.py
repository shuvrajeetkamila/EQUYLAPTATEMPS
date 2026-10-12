"""Pinned public benchmark acquisition and split hygiene. Never download model-generated answers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .audit import sha256_file

SOURCES = {
    "gsm8k": {
        "repository": "openai/grade-school-math",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "license": "MIT",
        "files": {
            "train": "grade_school_math/data/train.jsonl",
            "test": "grade_school_math/data/test.jsonl",
            "license": "LICENSE",
        },
    },
    "wikitext2": {
        "repository": "pytorch/examples",
        "revision": "acc295dc7b90714f1bf47f06004fc19a7fe235c4",
        "license": "CC-BY-SA-3.0 and GFDL (data), not the mirror's software license",
        "license_evidence": {
            "repository": "huggingface/datasets",
            "revision": "d7dfbc83d68e87ba002c5eb2555f7a932e59038a",
            "path": "datasets/wikitext/README.md",
        },
        "files": {
            "train": "word_language_model/data/wikitext-2/train.txt",
            "validation": "word_language_model/data/wikitext-2/valid.txt",
            "test": "word_language_model/data/wikitext-2/test.txt",
            "attribution": "word_language_model/data/wikitext-2/README",
        },
    },
}

# Measured from the pinned source Git blobs (not expected scores).
EXPECTED_SHA256 = {
    "gsm8k": {
        "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "license": "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b",
    },
    "wikitext2": {
        "train": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
        "validation": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
        "test": "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0",
        "attribution": "ed96062abea03d4f4e8ca80f0caad8b07ec59687b8064319c7ee6f6d900b6286",
    },
}


def github_json(endpoint: str) -> dict:
    """Use configured gh when available; public REST without credentials works in Colab."""
    if shutil.which("gh"):
        process = subprocess.run(
            ["gh", "api", endpoint], check=True, capture_output=True, text=True
        )
        return json.loads(process.stdout)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "heterogeneous-model-surgery"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request("https://api.github.com/" + endpoint, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def git_blob(repository: str, revision: str, path: str) -> bytes:
    """Fetch pinned Git blobs via api.github.com and verify their Git object identity."""
    tree = github_json(f"repos/{repository}/git/trees/{revision}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Truncated Git tree cannot establish a complete source inventory")
    entry = next((item for item in tree["tree"] if item["path"] == path), None)
    if entry is None or entry["type"] != "blob":
        raise ValueError(f"Pinned dataset file not found: {repository}:{path}")
    blob = github_json(f"repos/{repository}/git/blobs/{entry['sha']}")
    if blob.get("encoding") != "base64":
        raise ValueError("Dataset API did not return a base64 Git blob")
    content = base64.b64decode(blob["content"])
    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if digest != entry["sha"] or len(content) != entry.get("size", len(content)):
        raise ValueError("Dataset Git blob identity mismatch")
    return content


def acquire_benchmarks(target: str | Path) -> dict:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        manifest, _ = load_benchmarks(target)
        return manifest
    manifest = {"schema_version": 1, "sources": {}}
    for name, source in SOURCES.items():
        evidence = dict(source)
        evidence["local_files"] = {}
        directory = target / name
        directory.mkdir(exist_ok=True)
        for split, remote_path in source["files"].items():
            local = directory / Path(remote_path).name
            content = git_blob(source["repository"], source["revision"], remote_path)
            if local.exists() and local.read_bytes() != content:
                raise ValueError(f"Refusing changed cached benchmark file: {local}")
            local.write_bytes(content)
            evidence["local_files"][split] = {
                "path": str(local.resolve()),
                "sha256": sha256_file(local),
                "bytes": len(content),
            }
        if "license_evidence" in source:
            lic = source["license_evidence"]
            content = git_blob(lic["repository"], lic["revision"], lic["path"])
            local = directory / "source-dataset-card.md"
            local.write_bytes(content)
            evidence["license_evidence_sha256"] = sha256_file(local)
        manifest["sources"][name] = evidence
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_benchmarks(target: str | Path) -> tuple[dict, dict]:
    target = Path(target)
    manifest = json.loads((target / "manifest.json").read_text())
    data = {}
    if set(manifest["sources"]) != set(SOURCES):
        raise ValueError("Missing registered benchmark source")
    for name, source in manifest["sources"].items():
        data[name] = {}
        if source["revision"] != SOURCES[name]["revision"]:
            raise ValueError("Benchmark revision differs from registered source")
        for split, entry in source["local_files"].items():
            file = (target / name / Path(SOURCES[name]["files"][split]).name).resolve()
            if file != Path(entry["path"]).resolve():
                raise ValueError(
                    "Dataset manifest paths must remain in their declared source directory"
                )
            if (
                sha256_file(file) != entry["sha256"]
                or entry["sha256"] != EXPECTED_SHA256[name][split]
            ):
                raise ValueError("Benchmark checksum mismatch")
            if split in {"license", "attribution"}:
                continue
            if name == "gsm8k":
                rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
                for row in rows:
                    if (
                        set(row) != {"question", "answer"}
                        or not all(
                            isinstance(value, str) and value.strip() for value in row.values()
                        )
                        or "####" not in row["answer"]
                    ):
                        raise ValueError("Invalid GSM8K ground-truth schema")
                data[name][split] = rows
            else:
                data[name][split] = file.read_text()
    return manifest, data


def gsm_splits(
    rows: list[dict], *, seed: int, train_count: int, dev_count: int
) -> tuple[list, list]:
    """Select train/development from the published TRAIN file, never the benchmark test file."""
    if train_count < 1 or dev_count < 1 or train_count + dev_count > len(rows):
        raise ValueError("Invalid calibration/development sample counts")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = [dict(rows[i], source_index=i) for i in order[: train_count + dev_count]]
    train, dev = selected[:train_count], selected[train_count:]
    if {r["question"] for r in train} & {r["question"] for r in dev}:
        raise ValueError("GSM train/development prompt leakage")
    return train, dev


def text_chunks(text: str, *, characters: int = 192, limit: int = 64) -> list[str]:
    """Nonoverlapping character windows, not official word-level WikiText perplexity."""
    if characters < 16 or limit < 1:
        raise ValueError("Text chunk sizes must be positive (characters >= 16)")
    chunks = []
    for start in range(0, len(text), characters):
        chunk = text[start : start + characters]
        if chunk.strip():
            chunks.append(chunk)
        if len(chunks) == limit:
            break
    if not chunks:
        raise ValueError("No real text chunks found")
    return chunks
