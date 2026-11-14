"""Evaluate retrieved evidence and preserve per-query traces."""
import re
from typing import Any


def _normalized_text(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized)


def _gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = question.get("gold_source_id")
    page = question.get("gold_page")
    primary_quote = question.get("evidence_quote")

    evidence: list[dict[str, Any]] = []
    if primary_quote is not None:
        evidence.append(
            {"source_id": source_id, "page": page, "quote": primary_quote}
        )

    alternatives = question.get("evidence_alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError("evidence_alternatives must be a list")
    for alternative in alternatives:
        if isinstance(alternative, str):
            evidence.append(
                {"source_id": source_id, "page": page, "quote": alternative}
            )
        elif isinstance(alternative, dict):
            evidence.append(
                {
                    "source_id": alternative.get("gold_source_id", source_id),
                    "page": alternative.get("gold_page", page),
                    "quote": alternative.get("evidence_quote"),
                }
            )
        else:
            raise ValueError(
                "each evidence alternative must be a quote string or an object"
            )

    if not evidence:
        raise ValueError("question must include evidence_quote or evidence_alternatives")
    for item in evidence:
        if not isinstance(item["source_id"], str) or not item["source_id"].strip():
            raise ValueError("each gold evidence item must have a non-empty source ID")
        if not isinstance(item["page"], int) or isinstance(item["page"], bool):
            raise ValueError("each gold evidence item must have an integer page")
        if not isinstance(item["quote"], str) or not item["quote"].strip():
            raise ValueError("each gold evidence item must have a non-empty quote")
    return evidence


def _hit_matches(hit: dict[str, Any], evidence: dict[str, Any]) -> bool:
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("hit metadata must be an object")

    source = hit.get("document_id", metadata.get("document_id"))
    page = hit.get("page", metadata.get("page"))
    text = hit.get("text")
    if source != evidence["source_id"] or page != evidence["page"]:
        return False
    if not isinstance(text, str):
        raise ValueError("each hit must include text")
    return _normalized_text(evidence["quote"]) in _normalized_text(text)


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a trace with exact-evidence Hit@1 and Evidence Hit@3 labels.

    Questions may provide one or more ``evidence_alternatives``. Each item may
    be a quote string (same gold source/page) or an object containing
    ``evidence_quote`` and optional ``gold_source_id``/``gold_page`` overrides.
    """
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("each hit must be an object")

    evidence = _gold_evidence(question)
    hit_matches = [
        any(_hit_matches(hit, gold) for gold in evidence) for hit in hits
    ]
    hit1 = bool(hit_matches and hit_matches[0])
    hit3 = any(hit_matches[:3])

    return {
        "question_id": question.get("id"),
        "question": question.get("question"),
        "split": question.get("split", "unspecified"),
        "strategy": question.get("strategy", "unspecified"),
        "gold_evidence": evidence,
        "hits": hits,
        "hit_evidence": hit_matches,
        "hit_at_1": hit1,
        "evidence_hit_at_3": hit3,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 counts/rates by strategy and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trace in traces:
        strategy = trace.get("strategy", "unspecified")
        split = trace.get("split", "unspecified")
        if not isinstance(strategy, str) or not isinstance(split, str):
            raise ValueError("trace strategy and split must be strings")
        groups.setdefault((strategy, split), []).append(trace)

    summary: dict[str, Any] = {}
    for (strategy, split), group in sorted(groups.items()):
        total = len(group)
        hit1_count = sum(bool(trace.get("hit_at_1")) for trace in group)
        hit3_count = sum(bool(trace.get("evidence_hit_at_3")) for trace in group)
        summary.setdefault(strategy, {})[split] = {
            "query_count": total,
            "hit_at_1": {
                "hits": hit1_count,
                "total": total,
                "rate": hit1_count / total,
            },
            "evidence_hit_at_3": {
                "hits": hit3_count,
                "total": total,
                "rate": hit3_count / total,
            },
        }
    return summary
