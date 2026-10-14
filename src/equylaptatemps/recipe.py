"""Strict surgery recipes and an explicit, conservative parent-lineage allowlist."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Identity:
    family: str
    lineage: frozenset[str]
    evidence: str
    synthetic: bool = False


REGISTRY: dict[str, Identity] = {}
for size in ("130m", "370m", "790m", "1.4b", "2.8b"):
    REGISTRY[f"state-spaces/mamba-{size}-hf"] = Identity(
        "mamba", frozenset({"state-spaces/mamba"}), "https://github.com/state-spaces/mamba"
    )
for size in ("1.5B", "7B", "72B"):
    for suffix in ("", "-Instruct"):
        REGISTRY[f"Qwen/Qwen2.5-Math-{size}{suffix}"] = Identity(
            "qwen2", frozenset({"Qwen/Qwen"}), "https://github.com/QwenLM/Qwen2.5-Math"
        )
REGISTRY["HuggingFaceTB/SmolLM-135M"] = Identity(
    "llama", frozenset({"HuggingFaceTB/SmolLM"}), "https://huggingface.co/HuggingFaceTB/SmolLM-135M"
)
for family in ("mamba", "qwen2", "llama"):
    REGISTRY[f"synthetic://{family}"] = Identity(
        family, frozenset({f"random-initialization/{family}"}), "local random config", True
    )


def _keys(data: dict, expected: set[str], context: str) -> None:
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"{context} must contain exactly these keys: {sorted(expected)}")


@dataclass(frozen=True)
class ParentSpec:
    model_id: str
    local_path: str | None
    revision: str | None

    @property
    def identity(self) -> Identity:
        if self.model_id not in REGISTRY:
            raise ValueError(
                f"Unsupported parent {self.model_id!r}. Extend the reviewed provenance registry, "
                "not a flag that bypasses the stranger check."
            )
        return REGISTRY[self.model_id]

    def validate(self) -> None:
        if not isinstance(self.model_id, str):
            raise ValueError("model_id must be a string")
        identity = self.identity
        if self.local_path is not None and (
            not isinstance(self.local_path, str) or not self.local_path.strip()
        ):
            raise ValueError("local_path must be a nonempty string or null")
        if identity.synthetic:
            if self.revision is not None:
                raise ValueError("Synthetic parents have no pretrained revision; use null")
        else:
            if not isinstance(self.revision, str) or not re.fullmatch(
                r"[0-9a-f]{40}", self.revision
            ):
                raise ValueError(f"{self.model_id}: record an exact 40-hex parent revision")
            if len(set(self.revision)) == 1:
                raise ValueError("A repeated-character revision is a placeholder, not provenance")
            if self.local_path is None:
                raise ValueError(f"{self.model_id}: a local snapshot path is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: Path | None = None) -> ParentSpec:
        _keys(data, {"model_id", "local_path", "revision"}, "parent")
        if not isinstance(data["model_id"], str):
            raise ValueError("model_id must be a string")
        spec = cls(**data)
        spec.validate()
        if base is not None and spec.local_path is not None:
            spec = replace(spec, local_path=str((base / spec.local_path).resolve()))
        return spec


@dataclass(frozen=True)
class LayerSpan:
    """Half-open interval of complete blocks, never an arbitrary tensor slice."""

    start: int
    stop: int

    def validate(self, depth: int | None = None) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            raise ValueError("Layer bounds must be integers, not booleans or floats")
        if not 0 <= self.start < self.stop:
            raise ValueError("A sector must be a nonempty half-open interval [start, stop)")
        if depth is not None and self.stop > depth:
            raise ValueError(f"Sector [{self.start}, {self.stop}) exceeds parent depth {depth}")

    @classmethod
    def from_list(cls, value: list[int]) -> LayerSpan:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("A layer sector must be [start, stop]")
        span = cls(*value)
        span.validate()
        return span


@dataclass(frozen=True)
class Recipe:
    recipient: ParentSpec
    donor: ParentSpec
    recipient_cut: LayerSpan
    donor_sector: LayerSpan
    seed: int = 42
    schema_version: int = 1

    @property
    def synthetic(self) -> bool:
        return self.recipient.identity.synthetic or self.donor.identity.synthetic

    def validate(self, recipient_config=None, donor_config=None) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Only recipe schema_version 1 is supported")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.recipient.validate()
        self.donor.validate()
        a, b = self.recipient.identity, self.donor.identity
        if a.synthetic != b.synthetic:
            raise ValueError("Do not label a synthetic/pretrained mixture as a pretrained child")
        if a.family == b.family or a.lineage & b.lineage:
            raise ValueError(
                "Cousin parents rejected: families and known weight lineages must differ"
            )
        for spec, span, config in (
            (self.recipient, self.recipient_cut, recipient_config),
            (self.donor, self.donor_sector, donor_config),
        ):
            span.validate(None if config is None else config.num_hidden_layers)
            if config is not None and config.model_type != spec.identity.family:
                raise ValueError(
                    f"Parent {spec.model_id}: actual architecture {config.model_type!r} "
                    f"does not match declared family {spec.identity.family!r}"
                )
            if config is not None and getattr(config, "use_sliding_window", False):
                raise ValueError("Sliding-window Qwen2 attention is not supported by this runtime")

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: Path | None = None) -> Recipe:
        _keys(
            data,
            {"schema_version", "recipient", "donor", "recipient_cut", "donor_sector", "seed"},
            "recipe",
        )
        recipe = cls(
            recipient=ParentSpec.from_dict(data["recipient"], base),
            donor=ParentSpec.from_dict(data["donor"], base),
            recipient_cut=LayerSpan.from_list(data["recipient_cut"]),
            donor_sector=LayerSpan.from_list(data["donor_sector"]),
            seed=data["seed"],
            schema_version=data["schema_version"],
        )
        recipe.validate()
        return recipe

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), path.parent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipient": asdict(self.recipient),
            "donor": asdict(self.donor),
            "recipient_cut": [self.recipient_cut.start, self.recipient_cut.stop],
            "donor_sector": [self.donor_sector.start, self.donor_sector.stop],
            "seed": self.seed,
        }

    def stranger_report(self) -> dict[str, Any]:
        self.validate()
        return {
            "different_architectures": True,
            "disjoint_declared_weight_lineages": True,
            "lineage_evidence": {
                "recipient": self.recipient.identity.evidence,
                "donor": self.donor.identity.evidence,
            },
            "synthetic": self.synthetic,
            "limitation": (
                "Registry and local artifact checks cannot prove undisclosed weight ancestry "
                "or absence of shared training data. A revision is recorded, not remotely attested."
            ),
        }
