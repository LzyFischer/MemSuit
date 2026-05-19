"""
LongMemEval dataset — data structures and loader.

LongMemEval (Wu et al., ICLR 2025) tests five core long-term memory abilities
of chat assistants on multi-session histories. Each evaluation instance is
keyed by `question_id` and bundles:

    - the question + gold answer + question date
    - a list of timestamped chat sessions (the "haystack")
    - which sessions are evidence (used for retrieval-recall metrics; we
      don't use them when feeding the system, only for reporting)

We map the schema onto the same internal structures the LoCoMo evaluator
uses (Session / Turn / QA / Conversation / Sample), so the rest of the
pipeline (memory build → retrieval → answer generation) stays unchanged.

Mapping notes
-------------
- LongMemEval has one question per sample (vs. LoCoMo's many).
- "speakers" are roles, not names: "user" and "assistant". We feed both
  sides into the memory pipeline because some question types (e.g.
  `single-session-assistant`) explicitly test recall of assistant utterances.
- Each session has its own timestamp (haystack_dates[i]); we propagate that
  to every turn in the session, mirroring how LoCoMo's session_date_time
  is reused per turn.
- A category integer is assigned for compatibility with the existing
  metrics aggregator (which groups by integer category). The original
  string `question_type` is kept alongside for human-readable reporting.

Reference: https://github.com/xiaowu0162/LongMemEval (data format section)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


# ------------------------------------------------------------------
# Question types ↔ integer categories
# ------------------------------------------------------------------
#
# We assign stable integers so `aggregate_metrics(..., categories=...)` can
# still group by category. Abstention is its own category (mirrors LoCoMo
# cat 5) so adversarial-style branching keeps working.
QTYPE_TO_CAT: Dict[str, int] = {
    "single-session-user":        1,
    "single-session-assistant":   2,
    "single-session-preference":  3,
    "multi-session":              4,
    "temporal-reasoning":         5,
    "knowledge-update":           6,
    # 7 = abstention. We assign 7 dynamically when question_id ends in "_abs".
}
ABSTENTION_CAT = 7

CAT_TO_QTYPE: Dict[int, str] = {
    1: "single-session-user",
    2: "single-session-assistant",
    3: "single-session-preference",
    4: "multi-session",
    5: "temporal-reasoning",
    6: "knowledge-update",
    7: "abstention",
}

ABSTENTION_GOLD = "no information available"  # used only for token-overlap metrics


# ------------------------------------------------------------------
# Data structures (mirror eval/dataset.py)
# ------------------------------------------------------------------

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]  # answer_session_ids — session-level evidence
    category: Optional[int] = None
    question_id: Optional[str] = None       # stable id for hypothesis dumping
    question_type: Optional[str] = None     # original LongMemEval string
    question_date: Optional[str] = None
    is_abstention: bool = False

    @property
    def final_answer(self) -> Optional[str]:
        # For abstention, the gold answer is the system saying "I don't know"
        # in some form. Token-level metrics will mostly disagree on phrasing;
        # the LLM-judge is the metric of record for these. We use a stable
        # placeholder so metric code doesn't crash on `None`.
        if self.is_abstention:
            return self.answer or ABSTENTION_GOLD
        return self.answer


@dataclass
class Turn:
    speaker: str          # "user" or "assistant"
    dia_id: str           # synthetic: "S{session_idx}:T{turn_idx}"
    text: str
    has_answer: bool = False  # turn-level evidence flag from the dataset


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str        # "user"
    speaker_b: str        # "assistant"
    sessions: Dict[int, Session]


@dataclass
class LongMemEvalSample:
    """
    One LongMemEval instance. Schema-compatible with `LoCoMoSample`:
    `sample_id`, `qa`, `conversation`. `qa` has exactly one entry.
    """
    sample_id: str
    qa: List[QA]
    conversation: Conversation


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _parse_session(
    raw_session: List[dict],
    session_idx: int,
    session_date: str,
) -> Session:
    turns: List[Turn] = []
    for turn_idx, raw_turn in enumerate(raw_session):
        turns.append(
            Turn(
                speaker=raw_turn.get("role", "user"),
                dia_id=f"S{session_idx}:T{turn_idx}",
                text=raw_turn.get("content", ""),
                has_answer=bool(raw_turn.get("has_answer", False)),
            )
        )
    return Session(session_id=session_idx, date_time=session_date, turns=turns)


def _parse_instance(item: dict) -> LongMemEvalSample:
    qid: str = item["question_id"]
    qtype: str = item.get("question_type", "unknown")
    is_abs = qid.endswith("_abs")

    cat = ABSTENTION_CAT if is_abs else QTYPE_TO_CAT.get(qtype, 0)

    qa = QA(
        question=item["question"],
        answer=item.get("answer"),
        evidence=list(item.get("answer_session_ids", [])),
        category=cat,
        question_id=qid,
        question_type=qtype,
        question_date=item.get("question_date"),
        is_abstention=is_abs,
    )

    haystack_session_ids = item.get("haystack_session_ids", [])
    haystack_dates       = item.get("haystack_dates", [])
    haystack_sessions    = item.get("haystack_sessions", [])

    # Sanity: the three lists should be aligned. If misaligned (rare, but
    # the oracle file isn't sorted), we still iterate up to the shortest.
    n = min(len(haystack_session_ids), len(haystack_dates), len(haystack_sessions))

    sessions: Dict[int, Session] = {}
    for i in range(n):
        # Use the position in the list as our internal int session_id
        # (LongMemEval session ids are strings like "abc_sess_3" — we keep
        # the string in date_time / dia_id contexts via the Turn's dia_id
        # prefix, which uses positional indices for stable cross-references).
        sessions[i] = _parse_session(
            haystack_sessions[i],
            session_idx=i,
            session_date=haystack_dates[i],
        )

    conv = Conversation(
        speaker_a="user",
        speaker_b="assistant",
        sessions=sessions,
    )

    return LongMemEvalSample(
        sample_id=qid,
        qa=[qa],
        conversation=conv,
    )


# ------------------------------------------------------------------
# Public loader
# ------------------------------------------------------------------

def load_longmemeval(
    path: Union[str, Path],
    limit: Optional[int] = None,
) -> List[LongMemEvalSample]:
    """
    Load a LongMemEval JSON file (oracle / s / m / cleaned variants).

    Args:
        path:  path to the JSON file. The top-level is a list of 500
               evaluation instances.
        limit: if set, return only the first `limit` instances.

    Returns:
        List of `LongMemEvalSample`. Each has exactly one QA, which is
        what the evaluator iterates over.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected top-level list in {path}, got {type(raw).__name__}"
        )

    samples: List[LongMemEvalSample] = []
    for item in raw:
        try:
            samples.append(_parse_instance(item))
        except KeyError as e:
            print(f"[warn] skipping malformed instance "
                  f"(missing key {e}): {item.get('question_id', '?')}")
        if limit and len(samples) >= limit:
            break

    n_sessions = sum(len(s.conversation.sessions) for s in samples)
    n_turns    = sum(
        len(sess.turns)
        for s in samples
        for sess in s.conversation.sessions.values()
    )
    n_abs      = sum(1 for s in samples for q in s.qa if q.is_abstention)
    print(
        f"Loaded {len(samples)} LongMemEval instances "
        f"({n_sessions} sessions, {n_turns} turns total, {n_abs} abstention)"
    )
    return samples
