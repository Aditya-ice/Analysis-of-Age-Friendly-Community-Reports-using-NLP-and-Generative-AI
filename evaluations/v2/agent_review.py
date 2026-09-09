"""Offline session-agent feedback. No provider calls, activation, or gold-label writes."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid5

from elderhelp.config import Settings
from elderhelp.manifest import load_manifest
from elderhelp.v2.corpus import fingerprint, index_configuration
from elderhelp.v2.extraction import Block, ExtractedPage, prepare
from elderhelp.v2.identities import report_uuid
from integrity import digest
from review_batches import batches

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "evaluations/v2/agent-feedback"


def load():
    raw = (ROOT / "evaluations/v2/cases.jsonl").read_bytes()
    metadata = json.loads((ROOT / "evaluations/v2/dataset.json").read_text())
    if hashlib.sha256(raw).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Dataset changed; review version must be refreshed explicitly")
    cases = [c for line in raw.splitlines() if (c := json.loads(line))["split"] == "development"]
    catalog = {}
    config = fingerprint(index_configuration(Settings(_env_file=None)))
    if config != metadata["index_fingerprint"]:
        raise ValueError("Index configuration changed")
    for item in load_manifest(ROOT / "data/reports.yaml").reports:
        pdf = ROOT / item.local_path
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != item.expected_sha256:
            raise ValueError("Source PDF changed")
        pages = [
            ExtractedPage(**{**p, "blocks": [Block(**b) for b in p["blocks"]]})
            for p in json.loads((ROOT / f".local/extraction/{item.slug}.json").read_text())
        ]
        revision = uuid5(report_uuid(item.slug), item.expected_sha256)
        stored, _, spans, _ = prepare(revision, pages, config)
        numbers = {p.id: p.page_number for p in stored}
        for s in spans:
            alias = f"{item.slug}:{numbers[s.page_id]}:{s.start}"
            catalog[alias] = {
                "span_id": str(s.id),
                "revision_id": str(revision),
                "report_slug": item.slug,
                "pdf_page": numbers[s.page_id],
                "start": s.start,
                "end": s.end,
                "excerpt": s.text,
                "searchable": s.searchable,
                "confidence": s.confidence,
            }
    return cases, metadata, catalog


def validate(record, cases, metadata, catalog):
    expected = {c["id"]: c for c in cases}
    if (
        record.get("format") != "elderhelp-agent-feedback-v1"
        or record.get("review_type") != "ai_reviewed"
    ):
        raise ValueError("Explicit AI attribution required")
    if not all(record.get(k) for k in ("agent", "model", "method")):
        raise ValueError("Reviewer identity and method required")
    if datetime.fromisoformat(record["reviewed_at"]).tzinfo is None:
        raise ValueError("Review time must have a timezone")
    if (
        record.get("dataset_sha256") != metadata["dataset_sha256"]
        or record.get("index_fingerprint") != metadata["index_fingerprint"]
    ):
        raise ValueError("Stale dataset or extraction configuration")
    ids = [r["case_id"] for r in record["reviews"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Batch must contain exactly its development cases")
    spans = {s["span_id"]: s for s in catalog.values()}
    for row in record["reviews"]:
        case = expected[row["case_id"]]
        if row["case_sha256"] != digest(case):
            raise ValueError("Stale case feedback")
        if row["answerability"] not in (
            "full",
            "partial",
            "insufficient",
            "clarification_required",
        ):
            raise ValueError("Unknown answerability")
        if row["decision"] not in ("supported", "correction_needed", "unresolved"):
            raise ValueError("Unknown review decision")
        if not row["finding"] or not isinstance(row["limitations"], list):
            raise ValueError("Finding and limitations required")
        if row["answerability"] in ("full", "partial") and not row["supporting_spans"]:
            raise ValueError("Answerable finding needs evidence")
        for source in row["supporting_spans"]:
            if source != spans.get(source["span_id"]) or not source["searchable"]:
                raise ValueError("Unknown, altered, or unsearchable supporting span")
        if "gold_evidence_units" in row or "human_verified" in row:
            raise ValueError("AI feedback cannot certify gold or human review")


def save_batch(batch_number, annotations, output=DEFAULT_OUTPUT):
    cases, metadata, catalog = load()
    group = batches(cases)[batch_number - 1]
    path = output / f"development-{batch_number:02d}.json"
    if path.exists():
        validate(json.loads(path.read_text()), group, metadata, catalog)
        raise ValueError("Valid completed batch exists; preserve it instead of overwriting")
    record = {
        "format": "elderhelp-agent-feedback-v1",
        "review_type": "ai_reviewed",
        "agent": "Codex session agent",
        "model": "session model (exact ID unavailable)",
        "method": "direct_source_inspection",
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "dataset_sha256": metadata["dataset_sha256"],
        "index_fingerprint": metadata["index_fingerprint"],
        "batch": batch_number,
        "status": "AI-reviewed; manual spot-check pending",
        "reviews": [],
    }
    for case in group:
        entry = annotations[case["id"]]
        record["reviews"].append(
            {
                "case_id": case["id"],
                "case_sha256": digest(case),
                **{k: entry[k] for k in ("answerability", "decision", "finding", "limitations")},
                "supporting_spans": [catalog[k] for k in entry["sources"]],
                "visual_pages": entry.get("visual_pages", []),
            }
        )
    validate(record, group, metadata, catalog)
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    return path


def context(number):
    cases, _, catalog = load()
    group = batches(cases)[number - 1]
    print(
        json.dumps(
            [
                {
                    k: c[k]
                    for k in (
                        "id",
                        "category",
                        "question",
                        "history",
                        "expected_answerability",
                        "permitted_limitations",
                    )
                }
                for c in group
            ],
            indent=2,
        )
    )
    pages = {(s["report_slug"], s["pdf_page"]) for c in group for s in c["candidate_support"]}
    for alias, span in catalog.items():
        if (span["report_slug"], span["pdf_page"]) in pages:
            print(alias, "SEARCHABLE" if span["searchable"] else "UNCERTAIN", span["excerpt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["context", "save"])
    parser.add_argument("batch", type=int)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.command == "context":
        context(args.batch)
    else:
        print(save_batch(args.batch, json.loads(args.annotations.read_text())))
