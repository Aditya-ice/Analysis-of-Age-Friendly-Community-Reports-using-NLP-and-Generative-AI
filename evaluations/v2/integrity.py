"""Version binding and review completeness, shared by runners and release scoring."""

import hashlib
import json
from collections import Counter
from datetime import datetime

CATEGORIES = dict(
    direct=35,
    exact=15,
    comparison=15,
    followup=10,
    layout=10,
    unanswerable=20,
    partial=10,
    injection=5,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def reviewed(record, status_key="status"):
    try:
        return (
            record[status_key] == "human_verified"
            and bool(record["reviewer"].strip())
            and datetime.fromisoformat(record["reviewed_at"]).tzinfo is not None
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def audit(cases):
    errors = []
    if len(cases) != 120 or len({c["id"] for c in cases}) != 120:
        errors.append("Expected 120 unique case IDs")
    if Counter(c["category"] for c in cases) != CATEGORIES:
        errors.append("Category counts differ from the release protocol")
    if Counter(c["split"] for c in cases) != {"development": 80, "heldout": 40}:
        errors.append("Expected an 80/40 split")
    owners = {}
    for case in cases:
        keys = (
            ["topic:" + case["fact_group"]]
            + ["span:" + s["span_id"] for s in case["candidate_support"]]
            + [
                "span:" + s
                for u in case["gold_evidence_units"]
                for alt in u["alternatives"]
                for s in alt
            ]
        )
        for key in keys:
            if key in owners and owners[key] != case["split"]:
                errors.append(f"Cross-split leakage: {case['id']}, {key}")
            owners[key] = case["split"]
        if case["review"]["status"] == "pending":
            if case["gold_evidence_units"]:
                errors.append(f"{case['id']}: pending evidence must not be labeled gold")
            continue
        if not reviewed(case["review"]):
            errors.append(f"{case['id']}: incomplete human review identity/date")
        if (
            case["expected_answerability"] in ("full", "partial")
            and not case["gold_evidence_units"]
        ):
            errors.append(f"{case['id']}: answerable case needs reviewed evidence units")
        approved = {s["span_id"] for s in case["candidate_support"] if s["searchable"]}
        for unit in case["gold_evidence_units"]:
            if not unit.get("alternatives") or any(
                not a or not set(a) <= approved for a in unit["alternatives"]
            ):
                errors.append(
                    f"{case['id']}: gold alternatives need exact searchable reviewed spans"
                )
    return sorted(set(errors))
