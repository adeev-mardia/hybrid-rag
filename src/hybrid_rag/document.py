"""Shared document/result types used across the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        snippet = self.text[:60].replace("\n", " ")
        return f"ScoredDocument(id={self.doc_id!r}, score={self.score:.4f}, text={snippet!r}...)"
