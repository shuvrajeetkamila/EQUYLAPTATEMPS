"""Immutable experiment configurations and tamper-evident sequential campaign state."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .audit import sha256_file
from .campaign_training import stable_hash
from .evidence import checked_path


class CampaignRegistry:
    def __init__(self, root, experiments):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "campaign_02_results.json"
        configs = {e["id"]: e for e in experiments}
        if len(configs) != len(experiments):
            raise ValueError("Duplicate immutable experiment ID")
        self.configs = configs
        if self.file.exists():
            self.state = json.loads(self.file.read_text())
            if self.state["registry_sha256"] != stable_hash(experiments):
                raise ValueError("Changed campaign registry; create a new namespace")
        else:
            self.state = {
                "schema_version": 1,
                "project": "HETEROGENEOUS AI MODEL SURGERY",
                "campaign": 2,
                "registry_sha256": stable_hash(experiments),
                "records": {
                    e["id"]: {
                        "config_sha256": stable_hash(e),
                        "execution_status": "NOT_RUN",
                        "evaluation_status": "NOT_RUN",
                        "reason": "Not attempted",
                        "artifacts": {},
                    }
                    for e in experiments
                },
            }
            self.save()

    def save(self):
        temp = self.root / "campaign_02_results.new.json"
        temp.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        os.replace(temp, self.file)

    def verified_complete(self, id):
        r = self.state["records"][id]
        if r["config_sha256"] != stable_hash(self.configs[id]):
            raise ValueError("Experiment config identity changed")
        if r["execution_status"] != "COMPLETED":
            return False
        if not r["artifacts"]:
            raise ValueError("Completed experiment has no verified artifacts")
        for name, digest in r["artifacts"].items():
            if sha256_file(checked_path(self.root, name)) != digest:
                raise ValueError("Completed experiment artifact missing/tampered")
        return True

    def update(
        self,
        id,
        execution_status,
        *,
        evaluation_status="NOT_RUN",
        reason="",
        metrics=None,
        paths=(),
        metadata=None,
    ):
        if execution_status not in {
            "NOT_RUN",
            "RUNNING",
            "COMPLETED",
            "INTERRUPTED",
            "FAILED",
            "BLOCKED",
        }:
            raise ValueError("Unknown execution state")
        if evaluation_status not in {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}:
            raise ValueError("Unknown scientific evaluation state")
        artifacts = {
            str(Path(p).resolve().relative_to(self.root.resolve())): sha256_file(p) for p in paths
        }
        self.state["records"][id] = {
            "config_sha256": stable_hash(self.configs[id]),
            "execution_status": execution_status,
            "evaluation_status": evaluation_status,
            "reason": reason,
            "metrics": metrics,
            "artifacts": artifacts,
            "updated_clock_seconds": time.time(),
            "metadata": metadata or {},
        }
        self.save()

    def summary(self):
        return {
            "outcome": "NOT_CLASSIFIED",
            "reason": "Execution is not capability evidence; criteria decide promotion",
            "records": self.state["records"],
        }
