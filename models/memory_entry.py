"""
Core data structures: MemoryEntry (atomic unit) and Dialogue (raw input).

Paper ref: Section 3.1 — Atomic Entries {m_k}
  m_k = F_theta(W_t) = Phi_time ∘ Phi_coref ∘ Phi_extract(W_t)
  Indexed via: I(m_k) = {v_k (semantic)}  — semantic-only retrieval
"""
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MemoryEntry(BaseModel):
    """
    Atomic memory unit, self-contained and disambiguated.

    Fields:
      - entry_id: stable unique id for deduplication
      - lossless_restatement: the only stored content. Must be a complete,
        self-contained sentence (no pronouns, absolute timestamps inlined),
        because it is the only text we embed and the only text the answerer
        sees at retrieval time.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str


class Dialogue(BaseModel):
    """Raw input: a single conversational turn."""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None  # ISO 8601

    def __str__(self) -> str:
        prefix = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{prefix}{self.speaker}: {self.content}"
