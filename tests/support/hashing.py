# Copyright Xingyu Chen.
# Computes stable hashes for source bundles and packaged header sets.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_text_sha256(content: bytes) -> str:
    return sha256_bytes(content.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def header_set_sha256(headers: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for header in sorted(headers, key=lambda item: item["path"]):
        digest.update(header["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(header["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
