# Agent-assisted evidence review

**AI-reviewed; manual spot-check pending.** Completed September 9, 2026.

All 80 development cases were reviewed in their ten existing batches. The 40 held-out cases were excluded. Findings concern proposed evidence and answerability, not generated RAG answers or measured RAG accuracy.

## Outcomes

- 41 supported candidates, including supported scope/refusal expectations.
- 35 candidates requiring corrections to evidence selection, coverage, or interpretation.
- Four unresolved cases: three excluded infographic/chart cases and the request for exact per-participant costs across every program.
- Proposed answerability: 55 full, eight partial, 17 insufficient. These are agent judgments, not gold labels.

Each record contains the finding, limitations, exact supporting excerpts and span IDs, source revision, case hash, review timestamp, and AI attribution. See `evaluations/v2/agent-feedback/development-01.json` through `development-10.json`. The machine-readable summary includes category counts and visually inspected pages.

## What needs correction

Many original candidates contained headings or introductory material instead of substantive support. Several answers require the continuation on the following page, including rent-freeze outreach, fall prevention, aging-in-place guidance, and wheelchair dispatch. Comparison cases require evidence for both components; descriptions of mechanisms do not establish comparative effectiveness.

Historical commitments must remain historical. The housing target is not a count of completed units; proposed accessible-trip requirements are not evidence that every vehicle was accessible. NY Connects contacts are not unique people. Assistance with bills does not transfer financial decisions to volunteers. Reported survey importance is not participation, and general arts research does not establish that a particular program prevents dementia.

The NYC demographic infographics are visible in the original PDF but lack usable searchable spans in the current index. These cases remain unresolved rather than treating an extraction failure as evidence that the report lacks the information. Surprising chart labels were not silently corrected. Multi-column guide pages also need care where extracted reading order mixes captions and paragraphs.

Refusal findings distinguish corpus scope from exhaustive absence. Personal eligibility, medical decisions, live conditions, and unrelated communities cannot be established by these historical reports. Insufficient excerpts alone were not treated as proof of absence across the corpus. Prompt-injection cases record expected behavior; this review does not establish that a live attack was resisted.

## Method and limitations

The session agent inspected source files directly after browser access to the local review file was blocked. It inspected surrounding passages and rendered original PDF pages when numbers, tables, columns, or OCR mattered. No browser restriction was bypassed. Nineteen PDF pages were visually inspected; additional source text was read around candidate passages. Source hashes and supporting spans were validated against the local extraction and versioned configuration.

The recorded model is `session model (exact ID unavailable)`; no more specific model identity was asserted. Questions and report text were treated as untrusted evidence, not operational instructions. There were **zero Gemini/provider calls** in this workflow. The unfinished Gemini review script remains disabled outside it. No corpus pointer, deployment approval, original dataset, or human gold label was changed.

AI review can miss errors and cannot independently establish truth merely by agreeing with a report. It does not satisfy the human-reviewed release gate. Generated-answer verification, citation precision, supported-claim accuracy, live latency, and deployment remain later milestones.

## Twelve manual checks

Open `.local/evaluation-v2/agent-manual-spot-check.html`. It contains one case from each of the eight development categories, followed by four priority checks:

- `direct-01`, `exact-03`, `comparison-04`, `followup-07`, `layout-01`, `unanswerable-04`, `partial-01`, `injection-01`.
- `direct-19` (numerical transport interpretation), `layout-08` (excluded chart), `partial-05` (missing comparison component), `unanswerable-20` (ambiguous cost refusal).

Read each question, AI finding, limitations, and available exact excerpts. Follow the local source-page references where supplied. A refusal or missing-evidence case may have no supporting passage; that absence is stated rather than filled with invented evidence. Use the original reports for a broader check when needed. Leave review mode set to **Human**, record agreement or corrections, attest only to checks actually performed, and export the feedback. Human feedback remains separate from AI records. Currently **0/12 manual checks are completed** and no disagreements have yet been recorded.

## Reproduction and progress

With the existing local seed PDFs and extraction caches available:

```sh
PYTHONPATH=backend uv run python evaluations/v2/review_batches.py
PYTHONPATH=backend uv run python evaluations/v2/agent_report.py
```

The report command validates all saved batches before rebuilding the summary and local packet. Stale case, source, or configuration feedback is rejected; completed batches are preserved instead of overwritten. UI drafts bind to the complete packet and are rejected if stale. Browser draft persistence depends on local storage availability; exported feedback is the portable copy.

PDFs, caches, review-page renders, backups, and secrets are excluded from this delivery. The existing repository seed material is preserved. The committed findings contain attributed excerpts rather than full documents. Neither feedback export nor report generation activates the corpus or changes release approval.
