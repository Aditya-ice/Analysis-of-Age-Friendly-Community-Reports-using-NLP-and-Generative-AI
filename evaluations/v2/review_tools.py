"""Prepare human-review material. This tool never attests that an AI label is human-reviewed."""

import argparse
import hashlib
import html
import json
from pathlib import Path

from integrity import audit, digest, reviewed

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["packet", "annotations", "validate"])
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--output", type=Path, default=Path(".local/evaluation-v2"))
    parser.add_argument("--refresh-metadata", action="store_true")
    args = parser.parse_args()
    cases = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text().splitlines()]
    if args.command == "validate":
        errors = audit(cases)
        if errors:
            raise SystemExit("\n".join(errors))
        if args.refresh_metadata:
            metadata = json.loads((ROOT / "dataset.json").read_text())
            metadata["dataset_sha256"] = hashlib.sha256(
                (ROOT / "cases.jsonl").read_bytes()
            ).hexdigest()
            metadata["human_reviewed"] = sum(reviewed(c["review"]) for c in cases)
            (ROOT / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps({"valid": True, "human_reviewed": sum(reviewed(c["review"]) for c in cases)})
        )
    elif args.command == "packet":
        esc = html.escape
        content = [
            "<!doctype html><html lang=en><meta charset=utf-8>"
            "<title>ElderHelp evidence review</title>",
            "<style>body{font:18px system-ui;max-width:65rem;margin:2rem auto;padding:1rem}"
            "article{border-top:2px solid;padding:1rem 0}blockquote{white-space:pre-wrap}"
            "code{overflow-wrap:anywhere}summary{cursor:pointer}</style>",
            "<h1>Human evidence review — " + esc(args.split) + "</h1>",
            "<p>Candidate passages are suggestions, not gold labels. Check the original PDF, "
            "page reading order, exact offsets, answerability and every required component. "
            "Do not use held-out results to tune the system.</p>",
        ]
        for c in cases:
            if c["split"] != args.split:
                continue
            content.append(
                f"<article><h2>{esc(c['id'])}</h2><p>{esc(c['question'])}</p>"
                f"<pre>{esc(json.dumps(c['history'], indent=2))}</pre>"
                f"<p>Proposed status: {esc(c['expected_answerability'])}</p>"
                f"<p>{esc(json.dumps(c['required_components']))}</p>"
            )
            for s in c["candidate_support"]:
                content.append(
                    f"<details><summary>{esc(s['report_slug'])}, PDF page {s['pdf_page']} "
                    f"(searchable: {s['searchable']})</summary><code>{esc(s['span_id'])}</code>"
                    f"<blockquote>{esc(s['excerpt'])}</blockquote></details>"
                )
            content.append("</article>")
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / f"{args.split}-review.html"
        destination.write_text("\n".join(content) + "</html>\n")
        print(destination)
    else:
        destination = args.output / "annotations"
        destination.mkdir(parents=True, exist_ok=True)
        for path in args.output.glob("*.json"):
            trace = json.loads(path.read_text())
            if "displayed_claims" not in trace:
                continue
            target = destination / path.name
            if target.exists():
                continue
            value = dict(
                review_status="pending",
                reviewer=None,
                reviewed_at=None,
                trace_sha256=digest(trace),
                covered_components=[],
                injection_succeeded=None,
                claims=[
                    dict(
                        id=c["id"],
                        factual=None,
                        supported=None,
                        unsupported_date_quantity_or_current_service=None,
                        citation_support={s: None for s in c["citations"]},
                    )
                    for c in trace["displayed_claims"]
                ],
            )
            target.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
