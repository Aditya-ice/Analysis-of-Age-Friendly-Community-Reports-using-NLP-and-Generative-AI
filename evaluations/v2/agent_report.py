"""Publish offline AI findings and a separate manual spot-check; never alter labels."""

import json
from collections import Counter

from agent_review import DEFAULT_OUTPUT, ROOT, load, validate
from integrity import digest
from review_batches import batches, packet

SPOT_CHECK = [
    "direct-01",
    "exact-03",
    "comparison-04",
    "followup-07",
    "layout-01",
    "unanswerable-04",
    "partial-01",
    "injection-01",
    "direct-19",
    "layout-08",
    "partial-05",
    "unanswerable-20",
]


def load_agent_spot_check(cases_by_id, agent_reviews_by_id, metadata):
    path = DEFAULT_OUTPUT / "agent-manual-spot-check.json"
    if not path.exists():
        return 0
    record = json.loads(path.read_text())
    if any(key in record for key in ("human_verified", "gold_evidence_units", "reviewer")):
        raise ValueError("Agent spot check cannot carry human certification")
    if (
        record.get("format") != "elderhelp-agent-spot-check-v1"
        or record.get("review_type") != "ai_reviewed"
        or record.get("method") not in ("direct_source_inspection", "browser")
        or record.get("dataset_sha256") != metadata["dataset_sha256"]
        or not record.get("agent")
        or not record.get("model")
    ):
        raise ValueError("Invalid or stale agent spot check")
    reviews = record.get("reviews", [])
    if [review.get("case_id") for review in reviews] != SPOT_CHECK:
        raise ValueError("Agent spot check must cover the exact twelve selected cases")
    for review in reviews:
        case = cases_by_id[review["case_id"]]
        if review.get("case_sha256") != digest(case):
            raise ValueError("Agent spot check case is stale")
        evidence = (
            agent_reviews_by_id[review["case_id"]]["supporting_spans"] or case["candidate_support"]
        )
        expected_revisions = sorted({span["revision_id"] for span in evidence})
        if review.get("source_revision_ids") != expected_revisions:
            raise ValueError("Agent spot check source revision is stale")
        known_searchable = {span["span_id"] for span in evidence if span["searchable"]}
        if not set(review.get("selected_span_ids", [])) <= known_searchable:
            raise ValueError("Agent spot check selected an unknown or unsearchable span")
    return len(reviews)


def report():
    cases, metadata, catalog = load()
    records = []
    for number, group in enumerate(batches(cases), 1):
        record = json.loads((DEFAULT_OUTPUT / f"development-{number:02d}.json").read_text())
        validate(record, group, metadata, catalog)
        records.extend(record["reviews"])
    if len(records) != 80 or len({r["case_id"] for r in records}) != 80:
        raise ValueError("Expected 80 unique development reviews")
    by_id = {r["case_id"]: r for r in records}
    agent_spot_checks = load_agent_spot_check({case["id"]: case for case in cases}, by_id, metadata)
    categories = {c["id"]: c["category"] for c in cases}
    summary = {
        "status": (
            "AI-reviewed; agent spot-check complete; human spot-check pending"
            if agent_spot_checks == len(SPOT_CHECK)
            else "AI-reviewed; manual spot-check pending"
        ),
        "reviewed_cases": 80,
        "heldout_reviewed": 0,
        "manual_spot_checks_completed": 0,
        "agent_spot_checks_completed": agent_spot_checks,
        "provider_calls": 0,
        "quality_gate_passed": False,
        "dataset_sha256": metadata["dataset_sha256"],
        "decisions": dict(Counter(r["decision"] for r in records)),
        "answerability": dict(Counter(r["answerability"] for r in records)),
        "categories": {
            category: dict(
                Counter(r["decision"] for r in records if categories[r["case_id"]] == category)
            )
            for category in sorted(set(categories.values()))
        },
        "manual_spot_check_cases": SPOT_CHECK,
        "visual_source_pages": sorted({page for r in records for page in r["visual_pages"]}),
    }
    (DEFAULT_OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import yaml

    reports = {
        r["slug"]: r for r in yaml.safe_load((ROOT / "data/reports.yaml").read_text())["reports"]
    }
    selected = []
    for cid in SPOT_CHECK:
        original = next(c for c in cases if c["id"] == cid)
        review = by_id[cid]
        selected.append(
            {
                **original,
                "original_case_sha256": digest(original),
                "candidate_support": review["supporting_spans"] or original["candidate_support"],
                "agent_finding": review["finding"],
                "agent_limitations": review["limitations"],
            }
        )
    if len(set(SPOT_CHECK)) != 12 or len({c["category"] for c in selected[:8]}) != 8:
        raise ValueError("Spot-check must cover eight categories plus four distinct priority cases")
    html, payload = packet(selected, metadata, reports, "agent-manual-spot-check")
    destination = ROOT / ".local/evaluation-v2/agent-manual-spot-check.html"
    destination.write_text(html)
    lines = [
        "# Twelve-case manual spot-check",
        "",
        summary["status"],
        "",
        "Record human decisions separately; these findings do not certify gold labels.",
    ]
    for case in selected:
        review = by_id[case["id"]]
        lines.extend(
            [
                "",
                f"## {case['id']}",
                "",
                case["question"],
                "",
                f"AI finding ({review['answerability']}): {review['finding']}",
                "",
                "Limitations: " + "; ".join(review["limitations"]),
            ]
        )
        if not case["candidate_support"]:
            lines.extend(
                ["", "No supporting excerpt identified; assess scope and missing evidence."]
            )
        for source in case["candidate_support"]:
            lines.extend(
                [
                    "",
                    f"Source: {source['report_slug']}, PDF page {source['pdf_page']}; "
                    f"span {source['span_id']}.",
                    "",
                    "> " + source["excerpt"].replace("\n", "\n> "),
                ]
            )
        lines.extend(["", "Human decision: pending."])
    (DEFAULT_OUTPUT / "MANUAL_SPOT_CHECK.md").write_text("\n".join(lines) + "\n")
    (DEFAULT_OUTPUT / "manual-spot-check.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "case_ids": SPOT_CHECK,
                "dataset_sha256": metadata["dataset_sha256"],
                "agent_spot_checks_completed": agent_spot_checks,
                "human_spot_checks_completed": 0,
                "human_feedback": "pending",
                "selection": (
                    "One per category, then numerical claim, excluded chart, "
                    "missing comparison, ambiguous refusal"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(destination)


if __name__ == "__main__":
    report()
