"""
Evaluation metrics for memory QA.

Supported metrics:
  - exact_match
  - f1           (token-level)
  - rouge1/2/L   (F1)
  - bleu1-4
  - bert_f1      (BERTScore)
  - meteor
  - sbert        (sentence-transformers cosine similarity)
  - llm_judge    (LLM-as-a-judge, 0/1 score)
"""
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import nltk
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# Download NLTK data once
for pkg in ("punkt", "wordnet", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# SBERT model (lazy-loaded once)
_sbert_model: Optional[Any] = None


def _get_sbert() -> Optional[Any]:
    global _sbert_model
    if _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: SBERT model failed to load: {e}")
    return _sbert_model


# ------------------------------------------------------------------
# Individual metric functions
# ------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return (
        text.lower()
        .replace(".", " ").replace(",", " ")
        .replace("!", " ").replace("?", " ")
        .split()
    )


def token_f1(pred: str, ref: str) -> float:
    p_toks, r_toks = set(_tokenize(pred)), set(_tokenize(ref))
    common = p_toks & r_toks
    if not p_toks or not r_toks:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(r_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def rouge_scores(pred: str, ref: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge2_f": s["rouge2"].fmeasure,
        "rougeL_f": s["rougeL"].fmeasure,
    }


def bleu_scores(pred: str, ref: str) -> Dict[str, float]:
    p = nltk.word_tokenize(pred.lower())
    r = [nltk.word_tokenize(ref.lower())]
    smooth = SmoothingFunction().method1
    weights = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    out = {}
    for n, w in enumerate(weights, 1):
        try:
            out[f"bleu{n}"] = sentence_bleu(r, p, weights=w, smoothing_function=smooth)
        except Exception:
            out[f"bleu{n}"] = 0.0
    return out


def bert_f1_score(pred: str, ref: str) -> float:
    try:
        _, _, F1 = bert_score_fn([pred], [ref], lang="en", verbose=False)
        return F1.item()
    except Exception:
        return 0.0


def meteor(pred: str, ref: str) -> float:
    try:
        return meteor_score([ref.split()], pred.split())
    except Exception:
        return 0.0


def sbert_similarity(pred: str, ref: str) -> float:
    model = _get_sbert()
    if model is None:
        return 0.0
    try:
        e1 = model.encode([pred], convert_to_tensor=True)
        e2 = model.encode([ref], convert_to_tensor=True)
        return float(pytorch_cos_sim(e1, e2).item())
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# LLM judge
# ------------------------------------------------------------------

def llm_judge(
    pred: str, ref: str, question: str, judge_client: Any
) -> Tuple[float, str]:
    """
    Returns (score 0.0/1.0, reasoning string).
    Uses a relevance + accuracy rubric (generous on format variants).
    """
    if not pred or not ref:
        return 0.0, "empty prediction or reference"

    prompt = f"""You are an expert Relevance & Accuracy Evaluator.

Question: {question}
Reference Answer: {ref}
Predicted Answer: {pred}

Evaluation criteria:
1. Does the prediction contain the core factual content of the reference?
2. Partial, subset, or reformatted answers are acceptable (e.g. "2 PM" ≈ "14:00").
3. Score 1.0 if the prediction captures the key information; 0.0 only if clearly wrong.

Return JSON:
```json
{{"score": 1.0, "reasoning": "brief"}}
```
Return ONLY JSON."""

    messages = [
        {"role": "system", "content": "You are an expert evaluator. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        import config
        response_format = (
            {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
        )
        resp = judge_client.chat_completion(
            messages,
            temperature=getattr(config, "JUDGE_TEMPERATURE", 0.3),
            response_format=response_format,
            max_retries=3,
        )
        result = judge_client.extract_json(resp)
        return float(result.get("score", 0.0)), result.get("reasoning", "")
    except Exception as e:
        return 0.0, f"judge failed: {e}"


# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------

def compute_metrics(
    pred: str,
    ref: str,
    question: str = "",
    judge_client: Any = None,
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    if not pred or not ref:
        return _zero_metrics()

    pred, ref = str(pred).strip(), str(ref).strip()

    metrics: Dict[str, Any] = {
        "exact_match": int(pred.lower() == ref.lower()),
        "f1": token_f1(pred, ref),
        **rouge_scores(pred, ref),
        **bleu_scores(pred, ref),
        "bert_f1": bert_f1_score(pred, ref),
        "meteor": meteor(pred, ref),
        "sbert": sbert_similarity(pred, ref),
        "llm_judge": 0.0,
    }

    if use_llm_judge and question and judge_client:
        score, reasoning = llm_judge(pred, ref, question, judge_client)
        metrics["llm_judge"] = score
        metrics["llm_reasoning"] = reasoning

    return metrics


def _zero_metrics() -> Dict[str, Any]:
    return {
        "exact_match": 0, "f1": 0.0,
        "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
        "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
        "bert_f1": 0.0, "meteor": 0.0, "sbert": 0.0, "llm_judge": 0.0,
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, Any]],
    categories: List[int],
) -> Dict[str, Any]:
    """Compute mean/std/median across all metrics, split by QA category."""
    overall: Dict[str, List[float]] = defaultdict(list)
    by_cat: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for m, cat in zip(all_metrics, categories):
        for k, v in m.items():
            if isinstance(v, (int, float)):
                overall[k].append(float(v))
                by_cat[cat][k].append(float(v))

    def _stats(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
            "n": len(vals),
        }

    result: Dict[str, Any] = {
        "overall": {k: _stats(v) for k, v in overall.items() if v}
    }
    for cat in sorted(by_cat):
        result[f"cat_{cat}"] = {
            k: _stats(v) for k, v in by_cat[cat].items() if v
        }
    return result
