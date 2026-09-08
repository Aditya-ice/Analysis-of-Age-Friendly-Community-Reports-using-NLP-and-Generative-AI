"""Rebuild review candidates from the administrator's local extraction audit; never marks gold."""

import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid5

from elderhelp.config import Settings
from elderhelp.manifest import load_manifest
from elderhelp.v2.corpus import fingerprint, index_configuration
from elderhelp.v2.extraction import Block, ExtractedPage, prepare
from elderhelp.v2.identities import report_uuid

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("curation", Path(__file__).with_name("curation.py"))
curation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curation)


def build():
    destination = Path(__file__).with_name("cases.jsonl")
    if destination.exists() and any(
        json.loads(line)["review"]["status"] != "pending"
        for line in destination.read_text().splitlines()
        if line.strip()
    ):
        raise ValueError("Candidate rebuild would overwrite reviewed labels; create a new version")
    config = index_configuration(Settings(_env_file=None))
    manifests = {r.slug: r for r in load_manifest(ROOT / "data/reports.yaml").reports}
    sources = {}
    for slug, item in manifests.items():
        pages = [
            ExtractedPage(**{**p, "blocks": [Block(**b) for b in p["blocks"]]})
            for p in json.loads((ROOT / f".local/extraction/{slug}.json").read_text())
        ]
        revision = uuid5(report_uuid(slug), item.expected_sha256)
        stored, _, spans, _ = prepare(revision, pages, fingerprint(config))
        number = {p.id: p.page_number for p in stored}
        sources[slug] = (revision, spans, number)
    cases = []

    def add(
        category, group, question, references=(), answerability="full", history=None, missing=None
    ):
        evidence = []
        components = []
        for slug, page, anchor in references:
            revision, spans, numbers = sources[slug]
            words = set(re.findall(r"\w+", anchor.lower()))
            candidates = [s for s in spans if numbers[s.page_id] == page]
            candidates.sort(key=lambda s: (-sum(w in s.text.lower() for w in words), s.start))
            for span in candidates[:3]:
                evidence.append(
                    {
                        "report_slug": slug,
                        "revision_id": str(revision),
                        "span_id": str(span.id),
                        "pdf_page": page,
                        "start": span.start,
                        "end": span.end,
                        "excerpt": span.text,
                        "searchable": span.searchable,
                        "confidence": span.confidence,
                    }
                )
            components.append(
                f"Historical report evidence about {anchor} ({slug}, PDF page {page})"
            )
        if missing:
            components.append(missing)
        number = 1 + sum(c["category"] == category for c in cases)
        cases.append(
            {
                "id": f"{category}-{number:02d}",
                "category": category,
                "fact_group": group,
                "question": question,
                "history": history or [],
                "filters": {},
                "split": "pending",
                "expected_answerability": answerability,
                "required_components": components
                if category in ("comparison", "partial")
                else ([question] if answerability == "full" else []),
                "answerable_components": list(range(len(references)))
                if category in ("comparison", "partial")
                else ([0] if answerability == "full" else []),
                "permitted_limitations": [missing]
                if missing
                else (
                    ["Numerical infographic layout requires review"]
                    if group.startswith("figure_")
                    else []
                ),
                "expected_revision_ids": sorted({e["revision_id"] for e in evidence}),
                "candidate_support": evidence,
                "gold_evidence_units": [],
                "review": {"status": "pending", "reviewer": None, "reviewed_at": None},
            }
        )

    for category, rows in [
        ("direct", curation.DIRECT),
        ("exact", curation.EXACT),
        ("comparison", curation.COMPARISON),
    ]:
        for group, q, refs in rows:
            add(category, group, q, refs)
    for group, q, previous, refs in curation.FOLLOWUP:
        add("followup", group, q, refs, history=[{"role": "user", "content": previous}])
    for group, q, refs, status in curation.LAYOUT:
        add("layout", group, q, refs, status)
    for group, q in curation.UNANSWERABLE:
        add("unanswerable", group, q, answerability="insufficient")
    for group, q, refs, missing in curation.PARTIAL:
        add("partial", group, q, refs, "partial", missing=missing)
    for group, q in curation.INJECTION:
        add("injection", group, q, answerability="insufficient")
    assert len(cases) == 120
    # Merge same topical families AND shared candidate spans before assigning a split.
    parents = list(range(len(cases)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[root(b)] = root(a)

    owners = {}
    for i, case in enumerate(cases):
        keys = ["topic:" + case["fact_group"]] + [
            "span:" + e["span_id"] for e in case["candidate_support"]
        ]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i in range(len(cases)):
        groups[root(i)].append(i)
    groups = list(groups.values())
    random.Random(20260908).shuffle(groups)
    # Exact 40-case holdout, preserving complete related-fact groups.
    forced = next(
        group for group in groups if any(cases[i]["category"] == "injection" for i in group)
    )
    groups_to_split = [group for group in groups if group is not forced]
    reachable = {len(forced): forced.copy()}
    for group in groups_to_split:
        for count, path in sorted(list(reachable.items()), reverse=True):
            size = count + len(group)
            if size <= 40 and size not in reachable:
                reachable[size] = path + group
    if 40 not in reachable:
        raise ValueError("Fact groups cannot be split 80/40; revise curation before publishing")
    held = set(reachable[40])
    for i, case in enumerate(cases):
        case["split"] = "heldout" if i in held else "development"
    destination.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    schema = {
        "version": "research-v2.1",
        "index_configuration": config,
        "index_fingerprint": fingerprint(config),
        "dataset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "cases": len(cases),
        "human_reviewed": 0,
        "group_count": len(groups),
        "largest_group": max(map(len, groups)),
        "splits": dict(Counter(c["split"] for c in cases)),
        "categories": dict(Counter(c["category"] for c in cases)),
        "heldout_categories": dict(
            Counter(c["category"] for c in cases if c["split"] == "heldout")
        ),
    }
    Path(__file__).with_name("dataset.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    build()
