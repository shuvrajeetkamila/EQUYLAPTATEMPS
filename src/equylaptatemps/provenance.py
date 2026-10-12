"""Local snapshot evidence. Checksums are not cryptographic proof of undisclosed ancestry."""

from __future__ import annotations

import json
from pathlib import Path

from .audit import sha256_file
from .recipe import ParentSpec

SOURCE_RECORD = "surgery-source.json"


def snapshot_files(path: Path) -> dict[str, Path]:
    """Ignore Hub download caches, but account for every visible snapshot file."""
    return {
        file.relative_to(path).as_posix(): file
        for file in path.rglob("*")
        if file.is_file()
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
        and file != path / SOURCE_RECORD
    }


def snapshot_evidence(spec: ParentSpec, path: Path) -> dict:
    source = path / SOURCE_RECORD
    evidence = {"revision_evidence": "caller_declared_only", "signed_attestation": False}
    if source.exists():
        record = json.loads(source.read_text(encoding="utf-8"))
        if (
            type(record.get("format_version")) is not int
            or record["format_version"] != 1
            or record.get("model_id") != spec.model_id
        ):
            raise ValueError(f"Snapshot source record does not match parent {spec.model_id}")
        if record.get("revision") != spec.revision:
            raise ValueError("Snapshot source revision does not match the recipe")
        files = record.get("files")
        actual = snapshot_files(path)
        if not isinstance(files, dict) or set(files) != set(actual) or "config.json" not in files:
            raise ValueError("Snapshot inventory differs from its pinned download record")
        # Iterate actual local paths, never arbitrary filenames supplied by a source record.
        for relative, file in actual.items():
            if sha256_file(file) != files[relative]:
                raise ValueError(f"Snapshot checksum mismatch: {relative}")
        remote = record.get("hub_lfs_sha256", {})
        if not isinstance(remote, dict) or any(
            files.get(name) != value for name, value in remote.items()
        ):
            raise ValueError("Snapshot Hub LFS hash record does not match local inventory")
        evidence.update(
            {
                "hub_lfs_hashes_recorded": len(remote),
                "revision_evidence": "locally_verified_pinned_download_record",
                "source_record_sha256": sha256_file(source),
                "snapshot_files_sha256": files,
            }
        )
    notices = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.name.upper().startswith(("LICENSE", "NOTICE")):
            if file.stat().st_size > 1024 * 1024:
                raise ValueError(f"Unreasonably large parent license/notice document: {file.name}")
            notices[file.name] = {
                "sha256": sha256_file(file),
                "text": file.read_bytes().decode("utf-8"),
            }
    evidence["license_notices"] = notices
    evidence["redistribution_warning"] = (
        "Retained notices do not grant redistribution rights; inspect each parent license."
        if notices
        else "No parent license documents found. Resolve rights before redistributing weights."
    )
    return evidence
