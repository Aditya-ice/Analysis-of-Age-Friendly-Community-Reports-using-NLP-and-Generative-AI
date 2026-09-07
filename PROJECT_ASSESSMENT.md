# Elderhelp RAG — baseline assessment

Assessed September 6, 2026 (America/New_York). Baseline: `e48d9abf252eb4f1720e3ac5ae91e27a9b6f0c01`.

The GitHub connection is complete. The project is a useful research prototype with a small report corpus and a basic web interface, but it is not ready for deployment. Its main blockers are unreproducible dependencies, inconsistent file locations, legacy model configuration, missing index provisioning, and answers without visible evidence.

## Repository connection and preserved baseline

- Local folder: `/Users/adityamaheshwari/Documents/ChatGPT/Elderhelp RAG`.
- Origin (fetch and push): `https://github.com/Aditya-ice/Analysis-of-Age-Friendly-Community-Reports-using-NLP-and-Generative-AI.git`.
- Local `main` tracks `origin/main`; both point to the baseline SHA above.
- Current branch: `codex/project-assessment`, created from that same SHA.
- All nine existing commits and 13 original tracked files were fetched. `git fsck --full` passed.
- The local folder contained only its empty Git repository before checkout. No user files needed moving or replacement.
- Application code, reports, extracted text, and saved analysis remain unchanged. This assessment is the only project file added. No commit, push, ingestion, index creation, paid model call, or deployment was performed.

## Existing capabilities and reuse

| Component | Existing behavior | Reuse assessment |
| --- | --- | --- |
| Flask web app | `GET /` renders a question form; `POST /ask` accepts form field `question` and returns HTML | Preserve as a behavioral reference; frontend and backend choices belong to the remodel phase |
| Ingestion | Requests/BeautifulSoup scrape PDF links from one AARP member page | Keep as a starting adapter, after reliability and provenance improvements |
| Extraction | PyMuPDF direct extraction, then Tesseract OCR fallback | Reuse the direct-text-first approach; retain page metadata and apply OCR per page |
| NLP | scikit-learn LDA and spaCy organization/location counts; JSON export | Keep as exploratory analysis, not validated product insights |
| RAG | Text chunking, Google embeddings, local Chroma, LangChain RetrievalQA | Reuse the retrieval concept and sample questions; reassess implementation and package versions |
| Corpus | Two PDFs with matching extracted text files | Useful small regression corpus; insufficient for broad community coverage |

There are no public API changes in this phase. The existing app returns server-rendered HTML, not a JSON API. There is no report library UI, upload workflow, user account system, chat history, or deployment setup.

## Findings

### Startup, configuration, and deployment

1. **P0 — no reproducible installation.** The README requests `requirements.txt`, which does not exist. There is no package manifest, lockfile, or runtime configuration. Its clone example contains placeholder names and Markdown syntax inside the command; its script commands assume files at the repository root, while the pipeline lives under `age-friendly cities/`. The system Python lacks Flask and the core AI/OCR/NLP packages. With only Flask installed in a temporary virtual environment, the real app stops at `app.py:3` with `ModuleNotFoundError: No module named 'langchain_community'`.

2. **P0 — template capitalization breaks case-sensitive deployment.** `app.py:9` constructs Flask with its default `templates` directory, but Git tracks `Templates/index.html`. A case-sensitive template-lookup simulation raises `TemplateNotFound: index.html`. A case-insensitive macOS filesystem can conceal this problem. This was simulated locally, not tested on a Linux host.

3. **P0 — the pipeline and web app do not share a stable data root.** All paths are relative to the process working directory. From the repository root, `reports/` exists but `text_output/` does not. From the script directory, `text_output/` exists but `reports/` does not. Running the CLI there creates `age-friendly cities/db_gemini`, while the root-launched Flask app looks for `db_gemini` at the root. The analysis output location similarly changes with launch directory.

4. **P0 — index provisioning and validation are absent.** No index is included in the baseline. The Flask app opens Chroma but never ingests documents or checks collection size. The CLI treats directory existence as proof that an index is usable; it does not check collection contents, corpus changes, or embedding compatibility. An empty directory can bypass rebuilding. The exact empty-store behavior remains untested with real Chroma.

5. **P0 — model configuration needs replacement and centralization.** The web app uses `gemini-pro`; the CLI uses `gemini-2.0-flash`; both embed with `models/embedding-001`. Google's current schedule lists shutdown dates of June 1, 2026 for `gemini-2.0-flash` and October 30, 2025 for `embedding-001`. These dates have passed; the schedule notes dates are earliest possible retirement dates. This is documentary evidence, not an authenticated endpoint test. The exact `gemini-pro` alias was not verified. Choose supported models during implementation and rebuild embeddings rather than assuming an old index remains compatible. [Google Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations)

6. **P0 — startup is coupled to credentials and provider initialization.** The app exits during import if the key is absent, after dependency imports succeed. The no-key probe confirmed `SystemExit(None)`, which normally indicates a successful process exit despite failed startup. Other initialization exceptions set `qa_chain = None` and leave the web app running. The CLI prompts for a key during import, unsuitable for unattended startup.

7. **P1 — production operation is undefined.** The only executable web entry point uses `app.run(debug=True)`. There is no production server configuration, container, CI, health/readiness route, persistent storage plan, timeout policy, or deployment documentation. No application-level input length limit, rate limit, concurrency budget, or provider cost control is configured. Questions and answers are printed verbatim to logs. These are deployment requirements, not reasons to select a hosting vendor during this assessment.

### Answers and retrieval quality

- **P1 — citations are discarded.** Both chains request `return_source_documents=True`. The web route renders only `result`; the CLI also prints only that field. A mock answer containing a unique source marker displayed the answer but omitted the marker. The template has no source section.
- **P1 — page-level evidence cannot be reconstructed reliably from current output.** Extraction concatenates all pages into one text file without explicit page records. Plain-text chunking does not preserve document title, publisher, source URL, edition, or PDF page number. Printed numbers within text are not reliable structured page metadata.
- **P1 — error and input handling are weak.** Empty questions display an error, but whitespace-only questions reach the chain. Missing-chain and chain-exception cases display error messages with HTTP 200, making external health/error monitoring misleading.
- **P1 — answer quality has no measured baseline.** There is no evaluation set, retrieval relevance test, citation correctness check, or explicit application policy for insufficient evidence and conflicting reports. Model-generated answers and retrieval quality were not evaluated in this phase.

### Ingestion, extraction, and NLP

- The second `BASE_URL` assignment overrides the first, limiting ingestion to New York's AARP page. It does not crawl a community directory. PDF detection only accepts lowercase URLs ending exactly in `.pdf`, missing uppercase extensions and query-string URLs.
- Downloads have no timeout, retry policy, content validation, or atomic completion. A partially written file can be skipped on future runs merely because it exists. No source manifest records which link produced which bytes.
- OCR fallback uses a whole-document average of 100 characters per page. Mixed scanned/text documents may retain missing pages or trigger unnecessary OCR for every page. Existing text files are skipped without verifying whether their PDF changed. Exceptions are printed without an aggregate failure status; PDF resources are not closed on all error paths.
- The two NLP scripts largely duplicate each other. One imports unused pandas. Both require the separate spaCy `en_core_web_sm` model, which has no installation instructions, and truncate combined text at 500,000 characters. That limit is not reached by this corpus, whose extracted text totals 206,400 characters.
- The saved JSON has ten topics but only three distinct top-word lists: eight topics repeat exactly the same list. This is a verified property of the saved artifact, not a rerun of LDA. Ten topics on two full documents provides little basis for broad thematic claims. Entity output also contains split variants such as `New York City` and `New York \nCity`, and questionable location labels such as `FY`.
- The saved results lack a corpus hash, generation date, package/model versions, and processing configuration. Their exact reproducibility is unknown.

### Dependency inventory

Direct third-party requirements inferred from imports:

| Subsystem | Python packages and external assets |
| --- | --- |
| Web/RAG | Flask, langchain, langchain-community, langchain-google-genai, chromadb |
| Ingestion | requests, beautifulsoup4 |
| Extraction | PyMuPDF, pytesseract, Pillow; Tesseract executable and English OCR data |
| NLP | scikit-learn, spaCy, `en_core_web_sm`; pandas only for an unused import in script 3 |

This is an inventory, not a tested installation recipe. LangChain imports and the `DirectoryLoader` default loader's additional dependencies require resolution against a chosen version set. Tesseract was not found on PATH. No complete historical environment is available to recover exact versions.

## Corpus and provenance

| PDF | Evidence from existing extracted text | Current provenance confidence |
| --- | --- | --- |
| `AgeFriendlyNYC2017.pdf` | Title: *Age-friendly NYC: New Commitments for a City for All Ages*; NYC mayoral authorship on the cover; 158,579 extracted characters | The filename indicates 2017. No structured publication date or original download record exists |
| `engaging-community-single-pages-120318-lr.pdf` | Title: *Engaging the Community to Create Community*; AARP Network of Age-Friendly States and Communities publication; 2018 copyright notice; 47,821 extracted characters | A community-engagement guide with multi-city examples, not a second city's action plan |

The current AARP New York member page includes an action-plan download and the named community-engagement resource, supporting the collection context. Original download URLs, acquisition timestamps, and byte-for-byte matches to publisher files were not verified; no documents were downloaded in this phase. [AARP New York member page](https://livablemap.aarp.org/member/new-york-ny)

Baseline PDF SHA-256 fingerprints:

```text
AgeFriendlyNYC2017.pdf
19b42338e37d9899d18f71658968c33a0af486e7f5db5ec107edba4fd4bf5946

engaging-community-single-pages-120318-lr.pdf
2c436df93efdcd9a5b93c88720641e068a1a5fe77b44f8e263addeb69b5679ef
```

Keep publisher attribution and edition dates visible in a remodeled product. The repository has no license file or recorded redistribution terms for the bundled reports; the AARP text contains an all-rights-reserved notice. Record the applicable reuse basis before deciding whether to serve full PDFs publicly. This assessment makes no legal determination. Treat the corpus as historical material rather than proof of currently available services.

## Verification performed and limits

Tests ran with Python 3.11.0 in `/private/tmp/elderhelp-assessment.bCabNO/venv`, installing only Flask 3.1.2 and its dependencies. The temporary probe is `/private/tmp/elderhelp-assessment.bCabNO/probe.py`. Its network connections are blocked, provider/vector-store classes are test doubles, and it uses a dummy key after testing the absent-key path. It imports the original app without editing it. Temporary files are not part of the project.

| Check | Result |
| --- | --- |
| Origin, tracking, baseline SHA, nine commits, Git integrity | Passed |
| Parse all six Python files without executing them | Passed |
| Real startup with Flask-only environment | Blocked at missing `langchain_community` |
| Missing key after mocking dependency imports | Confirmed `SystemExit(None)` |
| Case-sensitive template lookup simulation | Confirmed `TemplateNotFound` |
| Home route with original template supplied in memory | HTTP 200 |
| Missing question | Error message present; HTTP 200 |
| Mock answer and returned source | Answer present; source absent; HTTP 200 |
| Whitespace-only question | Reached the mock chain |
| Unavailable chain and simulated chain exception | Error messages present; HTTP 200 in both cases |
| Saved JSON and path inventory | Confirmed duplicate topics and split data locations |
| Original tracked-file diff | Empty after assessment checks |

The Flask route checks isolate application behavior; they do not establish compatibility with real LangChain, Chroma, Google SDKs, or a production Linux filesystem. Full RAG startup, provider authentication, embedding generation, retrieval quality, OCR accuracy, NLP reproducibility, scraper download behavior, and deployment remain unverified. No credentials were requested or printed.

Temporary probe rerun command (available until temporary files are cleaned up):

```sh
/private/tmp/elderhelp-assessment.bCabNO/venv/bin/python -I /private/tmp/elderhelp-assessment.bCabNO/probe.py
```

## Prioritized modernization backlog

These are recommendations for the next phase, not changes made by this assessment.

| Order | Work package | Acceptance criteria |
| --- | --- | --- |
| 1 — P0 | Reproducible baseline | Choose and lock a supported Python/package set; document model/OCR assets and environment variables; repair setup instructions; clean install and import checks pass |
| 2 — P0 | Stable startup and paths | Use a shared explicit data root; fix template casing; validate configuration without interactive prompts; Linux route smoke tests pass; launching from another directory uses identical data |
| 3 — P0 | Supported RAG and index lifecycle | Centralize model settings; select available models; separate indexing from serving; verify collection contents and corpus/embedding versions; missing or stale index produces actionable readiness failure |
| 4 — P1 | Evidence-backed answers | Preserve report/page/source metadata during extraction; show citations with answers; add insufficient-evidence handling; evaluate known-answer, absent-answer, conflicting-report, and citation cases |
| 5 — P1 | Reliable corpus processing | Add a source manifest, hashes, atomic downloads, timeouts, and per-page OCR; detect changed inputs; test corrupt PDFs, mixed scans, repeat runs, and interrupted downloads |
| 6 — P1 | Product and deployment definition | Agree target users, first-release workflows, corpus scope, and cost ceiling; then select UI and hosting architecture with persistent index storage and an explicit report-publication policy |
| 7 — P1 | Deployment readiness | Add production serving, health/readiness checks, input/rate/cost controls, appropriate status codes, structured logs, CI, and rollback instructions; verify restart persistence and provider-outage behavior in staging |
| 8 — P2 | Reassess analytics | Consolidate duplicate scripts; normalize entities; select a topic-analysis approach suitable for the corpus; store provenance and evaluate outputs before exposing charts or claims |

Recommended next milestone: a reproducible local app answering questions over the two preserved reports with verifiable page citations. Broader corpus expansion, visual redesign, provider selection, and hosting are deliberately left for the product-remodel phase.
