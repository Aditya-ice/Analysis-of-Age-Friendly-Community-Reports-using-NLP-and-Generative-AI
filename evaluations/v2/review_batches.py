"""Offline development review packs. Exported feedback never changes gold labels."""

import argparse
import hashlib
import html
import json
from collections import OrderedDict
from pathlib import Path

import yaml
from integrity import audit, digest

ROOT = Path(__file__).resolve().parents[2]


def batches(cases, size=10):
    """Keep related facts AND shared spans together; never emit held-out content."""
    if size < 1:
        raise ValueError("Batch size must be positive")
    development = [c for c in cases if c["split"] == "development"]
    parents = list(range(len(development)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, case in enumerate(development):
        keys = ["fact:" + case["fact_group"]] + [
            "span:" + s["span_id"] for s in case["candidate_support"]
        ]
        for key in keys:
            if key in owners:
                parents[root(i)] = root(owners[key])
            owners[key] = i
    groups = OrderedDict()
    for i, case in enumerate(development):
        groups.setdefault(root(i), []).append(case)
    result, current = [], []
    for group in groups.values():
        if current and len(current) + len(group) > size:
            result.append(current)
            current = []
        current.extend(group)
    if current:
        result.append(current)
    return result


STYLE = (Path(__file__).parent / "review_ui.css").read_text()
SCRIPT = (Path(__file__).parent / "review_ui.mjs").read_text()


def packet(cases, metadata, reports, batch_id):
    esc = html.escape
    payload = {
        "batch_id": batch_id,
        "dataset_sha256": metadata["dataset_sha256"],
        "index_fingerprint": metadata["index_fingerprint"],
        "cases": [
            {
                "id": c["id"],
                "sha256": digest(c),
                "source_revision_ids": sorted({s["revision_id"] for s in c["candidate_support"]}),
                "searchable_span_ids": [
                    s["span_id"] for s in c["candidate_support"] if s["searchable"]
                ],
            }
            for c in cases
        ],
    }
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>ElderHelp review — {esc(batch_id)}</title><style>{STYLE}</style></head><body>",
        f"<h1>Evidence review — {esc(batch_id)}</h1>",
        '<p class="notice">Development cases only. These are candidate labels, not gold evidence. '
        "Check the original PDF and its reading order, not only the extracted passage. "
        "This local page makes no API calls. Drafts are saved in this browser when available. "
        "Export before closing.</p>",
        "<p>Select the passages that support each answer component. Explain missing evidence, "
        "uncertain tables, or needed corrections in the notes. Different statements may need "
        "different passages; the final evidence-unit grouping will be checked "
        "before labels change.</p>",
        '<label>Your name <input id="reviewer" type="text" autocomplete="name"></label>',
        '<label>Reviewer type <select id="review-mode"><option value="human">Human</option>'
        '<option value="ai">AI agent (not human verification)</option></select></label>'
        '<label>AI model identity <input id="model" type="text"></label>'
        '<label>AI review method <select id="method"><option value="direct_source_inspection">'
        'Direct source inspection</option><option value="browser">Browser</option>'
        "</select></label>",
    ]
    for c in cases:
        pieces += [
            f'<article id="{esc(c["id"])}"><h2>{esc(c["id"])}: {esc(c["question"])}</h2>',
            f"<p>Category: {esc(c['category'])} · Proposed answerability: "
            f"{esc(c['expected_answerability'])}</p>",
        ]
        if c["history"]:
            pieces.append(f"<pre>{esc(json.dumps(c['history'], indent=2))}</pre>")
        pieces.append("<h3>Proposed required components</h3><ul>")
        pieces.extend(f"<li>{esc(str(component))}</li>" for component in c["required_components"])
        pieces.append("</ul>")
        if c.get("filters"):
            pieces.append(f"<p>Report filters: {esc(json.dumps(c['filters']))}</p>")
        if c.get("permitted_limitations"):
            pieces.append(f"<p>Proposed limitations: {esc(str(c['permitted_limitations']))}</p>")
        if not c["candidate_support"]:
            pieces.append(
                "<p>No candidate supporting passage. Check the question against the "
                "approved corpus before confirming insufficient evidence.</p>"
            )
        for span in c["candidate_support"]:
            report = reports[span["report_slug"]]
            pdf = (ROOT / report["local_path"]).resolve()
            pdf.relative_to((ROOT / "data/seed/pdfs").resolve())
            pieces.append(
                f"<details><summary>{esc(report['title'])} · PDF page {span['pdf_page']}</summary>"
                f'<p><a href="{esc(pdf.as_uri())}#page={span["pdf_page"]}">Open original local PDF '
                f"at page {span['pdf_page']}</a></p><p>If your viewer blocks local links, open "
                f"{esc(pdf.name)} manually at that PDF page.</p>"
                f"<code>Source span: {esc(span['span_id'])}</code>"
                f"<p>Revision: <code>{esc(span['revision_id'])}</code> · "
                f"Normalized offsets: {span['start']}–{span['end']} · "
                f"Extraction confidence: {span['confidence']}</p>"
                f"<blockquote>{esc(span['excerpt'])}</blockquote>"
            )
            if span["searchable"]:
                pieces.append(
                    f'<label><input class="span" type="checkbox" value="{esc(span["span_id"])}"> '
                    "This passage supports an answer component</label>"
                )
            else:
                pieces.append(
                    "<p>Not searchable: uncertain extraction. Do not approve this as "
                    "retrieval gold evidence; note any extraction correction needed.</p>"
                )
            pieces.append("</details>")
        pieces += [
            '<label>Decision <select class="decision"><option value="pending">Pending</option>'
            '<option value="candidate_supported">Candidate checked and supported</option>'
            '<option value="changes_needed">Changes or more evidence needed</option>'
            "</select></label>",
            '<label>Your answerability assessment <select class="answerability">'
            '<option value="">Choose after checking the source</option>'
            '<option value="full">Fully answerable</option>'
            '<option value="partial">Partially answerable</option>'
            '<option value="insufficient">Insufficient evidence</option>'
            '<option value="clarification_required">Clarification required</option>'
            "</select></label>",
            '<label><input class="source-checked" type="checkbox"> I checked the original source '
            "and all required answer components for this case</label>",
            "<label>Corrections, evidence grouping, or limitations "
            "<textarea></textarea></label></article>",
        ]
    pieces += [
        '<label><input id="attest" type="checkbox"><span id="attestation-text"> '
        "I am the named human reviewer and personally "
        "checked the original sources for the cases I am submitting.</span></label>",
        '<button id="export" type="button">Export reviewed cases as feedback</button>'
        '<p id="status" role="status" aria-live="polite"></p>',
        "<p>Exporting does not update the dataset, approve deployment, or run Google calls. "
        "Feedback is bound to these exact case versions. Keep it private until reviewed.</p>",
        '<script id="packet" type="application/json">'
        + json.dumps(payload).replace("<", "\\u003c")
        + "</script>",
        f'<script type="module">{SCRIPT}</script></body></html>',
    ]
    return "\n".join(pieces), payload


def generate(output, size=10):
    source = ROOT / "evaluations/v2/cases.jsonl"
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    metadata = json.loads((source.parent / "dataset.json").read_text())
    if (
        audit(cases)
        or hashlib.sha256(source.read_bytes()).hexdigest() != metadata["dataset_sha256"]
    ):
        raise ValueError("Dataset integrity or version check failed; do not prepare stale packets")
    reports = {
        r["slug"]: r for r in yaml.safe_load((ROOT / "data/reports.yaml").read_text())["reports"]
    }
    destination = output / metadata["dataset_sha256"][:12]
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for i, group in enumerate(batches(cases, size), 1):
        batch_id = f"development-{i:02d}"
        document, payload = packet(group, metadata, reports, batch_id)
        path = destination / f"{batch_id}.html"
        path.write_text(document)
        records.append({**payload, "file": path.name, "case_count": len(group)})
    index = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>ElderHelp review batches</title>"
    )
    index += f"<style>{STYLE}</style><h1>Development evidence review</h1><p>80 development cases; "
    index += (
        "held-out material is excluded. Related facts and shared source spans "
        "stay together.</p><ol>"
    )
    for item in records:
        index += (
            f'<li><a href="{item["file"]}">{item["batch_id"]}</a> — {item["case_count"]} cases</li>'
        )
    index += "</ol><p>No label is marked human-reviewed by this tool.</p></html>"
    (destination / "index.html").write_text(index)
    (destination / "batches.json").write_text(json.dumps(records, indent=2) + "\n")
    return {
        "index": str(destination / "index.html"),
        "batches": len(records),
        "development_cases": sum(r["case_count"] for r in records),
        "heldout_cases_exported": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/evaluation-v2/review-batches"))
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
