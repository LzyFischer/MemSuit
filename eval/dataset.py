"""
LoCoMo dataset — data structures and loader.

Extracted from the original test_locomo10.py to keep evaluation
concerns separate from data parsing.

QA categories:
  1 = single-hop      3 = temporal
  2 = multi-hop       4 = commonsense    5 = adversarial
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Ground-truth answer: adversarial_answer for cat-5, else answer."""
        return self.adversarial_answer if self.category == 5 else self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class LoCoMoSample:
    sample_id: str
    qa: List[QA]
    conversation: Conversation


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _parse_turn(turn_data: dict) -> Turn:
    text = turn_data.get("text", "")
    if "img_url" in turn_data and "blip_caption" in turn_data:
        caption = f"[Image: {turn_data['blip_caption']}]"
        text = f"{caption} {text}".strip() if text else caption
    return Turn(
        speaker=turn_data["speaker"],
        dia_id=turn_data["dia_id"],
        text=text,
    )


def _parse_session(session_data: list, session_id: int, date_time: str) -> Session:
    turns = [_parse_turn(t) for t in session_data]
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def _parse_conversation(conv: dict) -> Conversation:
    sessions: Dict[int, Session] = {}
    for key, value in conv.items():
        if key.startswith("session_") and isinstance(value, list):
            sid = int(key.split("_")[1])
            dt = conv.get(f"{key}_date_time")
            if dt:
                session = _parse_session(value, sid, dt)
                if session.turns:
                    sessions[sid] = session
    return Conversation(
        speaker_a=conv["speaker_a"],
        speaker_b=conv["speaker_b"],
        sessions=sessions,
    )


# ------------------------------------------------------------------
# Public loader
# ------------------------------------------------------------------

def load_locomo(path: Union[str, Path], limit: Optional[int] = None) -> List[LoCoMoSample]:
    """
    Load LoCoMo JSON dataset.

    Args:
        path:  path to locomo10.json (or full locomo.json)
        limit: if set, return only the first `limit` samples
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    samples: List[LoCoMoSample] = []
    for idx, item in enumerate(raw):
        qa_list = [
            QA(
                question=qa["question"],
                answer=qa.get("answer"),
                evidence=qa.get("evidence", []),
                category=qa.get("category"),
                adversarial_answer=qa.get("adversarial_answer"),
            )
            for qa in item["qa"]
        ]
        samples.append(
            LoCoMoSample(
                sample_id=str(idx),
                qa=qa_list,
                conversation=_parse_conversation(item["conversation"]),
            )
        )
        if limit and len(samples) >= limit:
            break

    print(
        f"Loaded {len(samples)} samples, "
        f"{sum(len(s.qa) for s in samples)} QA pairs total"
    )
    return samples
