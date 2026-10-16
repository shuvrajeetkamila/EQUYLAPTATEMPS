"""The child owns exactly one vocabulary: the recipient's. Donor IDs are never reused."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...
    def save_pretrained(self, path: str): ...


class ByteTokenizer:
    """Offline fixture only; UTF-8 byte IDs, optionally permuted for a distinct donor vocabulary."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    vocab_size = 259

    def __init__(self, rotation: int = 0):
        if type(rotation) is not int or not 0 <= rotation < 256:
            raise ValueError("Byte rotation must be in [0, 256)")
        self.rotation = rotation

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [3 + (byte + self.rotation) % 256 for byte in text.encode("utf-8")]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        # Invalid/random byte sequences in a synthetic model are explicitly replacement-decoded.
        return bytes((i - 3 - self.rotation) % 256 for i in ids if 3 <= i < 259).decode(
            "utf-8", errors="replace"
        )

    def get_vocab(self) -> dict[str, int]:
        return {
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
            **{f"byte:{i}": 3 + (i + self.rotation) % 256 for i in range(256)},
        }

    def save_pretrained(self, path: str):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "byte_tokenizer.json").write_text(
            json.dumps({"kind": "synthetic_utf8_bytes", "rotation": self.rotation}, indent=2)
            + "\n",
            encoding="utf-8",
        )


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    fixture = path / "byte_tokenizer.json"
    if fixture.exists():
        data = json.loads(fixture.read_text(encoding="utf-8"))
        if data.get("kind") != "synthetic_utf8_bytes" or set(data) != {"kind", "rotation"}:
            raise ValueError("Invalid byte tokenizer fixture")
        return ByteTokenizer(data["rotation"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Recipient/donor tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        # Reuse an existing ID: adding a new token would change the frozen embedding contract.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special_ids": [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id],
    }
    if hasattr(tokenizer, "backend_tokenizer"):
        payload["backend"] = json.loads(tokenizer.backend_tokenizer.to_str())
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_tokenizer(tokenizer: Tokenizer, vocab_size: int) -> None:
    ids = list(tokenizer.get_vocab().values())
    ids += [tokenizer.pad_token_id, tokenizer.eos_token_id]
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    if not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError(
            "Tokenizer IDs do not fit the parent's frozen vocabulary; no resizing allowed"
        )


ANSWER_SEPARATOR = "\n\nAnswer:"


def prompt_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
    # Encoding the same prefix at train and inference time prevents answer-boundary retokenization.
    return tokenizer.encode(prompt + ANSWER_SEPARATOR, add_special_tokens=False)
