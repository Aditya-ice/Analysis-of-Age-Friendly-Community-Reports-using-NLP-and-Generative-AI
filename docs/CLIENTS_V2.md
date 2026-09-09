# Pilot clients and shared contract

The browser, SwiftUI app and Compose app use `/v2`. Pilot access is entered at runtime; no live invite code or Google credential is bundled. Tokens stay in browser session storage, iOS device-only Keychain and Android Keystore-encrypted preferences. Android backup remains disabled. Cached v1 histories remain readable; missing legacy revision/span provenance is treated as unknown, while newly received v2 answers require provenance.

All clients show verification progress and display the completed verified response. They distinguish partial/insufficient/clarification results and show exact excerpts, publisher links, report dates and printed page labels. Keyword search remains a separate action. New questions send no saved history by default; the explicit follow-up control uses at most six turns from the active in-memory conversation. Individual history deletion and clearing are local operations. Unfinished answers are never saved.

The browser is served by FastAPI at `/`, with an allowlisted static-asset route and restrictive CSP. It has no third-party scripts or analytics. IndexedDB stores completed answers and report metadata. The service worker caches only the application shell, never API responses, questions, tokens or PDFs. Native metadata is stored in SwiftData/Room. Access expiry, quota limits and connection/sleep failures have distinct messages; retries require a user action. Publisher pages open externally and may themselves link to PDFs.

All three interfaces disclose Google's handling of questions/evidence and restrict the intended audience to non-sensitive research. The [Google unpaid-service terms](https://ai.google.dev/gemini-api/terms), [SwiftUI accessibility guidance](https://developer.apple.com/documentation/swiftui/accessibility-fundamentals) and [Android architecture recommendations](https://developer.android.com/topic/architecture/recommendations) inform this implementation. Controls use text labels, large touch areas and non-color status messages. Browser layout supports narrow screens, high contrast and reduced motion; SwiftUI uses scalable system text and Compose uses native semantics.

## Fixture verification

`contracts/fixtures/` is the shared contract source. Test-only copies are generated for native bundles with `python3 tools/sync_contract_fixtures.py`; Kotlin network models come from `python3 android/tools/generate_models.py`. No test fixture or mock credential enters a published app bundle.

```sh
python3 tools/pilot_mock.py
# Open http://127.0.0.1:8765. Fixture access code: mock-invite-code-only
node --test web-tests/core.test.mjs
```

The server binds to loopback, performs no Google calls and returns explicitly synthetic client fixtures. Questions `quota`, `failure`, and `slow` exercise limits, verification failure and cancellation. It is not the real RAG engine. The original iOS mock-server entry point delegates to this server so both native CI jobs share the same events.

Set the iOS simulator environment `ELDERHELP_API_BASE_URL=http://localhost:8765`. For Android emulator builds use `-PelderhelpApiUrl=http://10.0.2.2:8765/`; only the debug manifest allows cleartext test traffic. Production URLs must use HTTPS.

Current checks: browser protocol tests pass, and the browser invite → partial answer → exact citation → report library → saved history flow was inspected against the fixture server. The static-serving regression confirms backend code and PDFs cannot be downloaded through asset routes. Swift core compiles locally, but this machine's command-line tools have no XCTest or iOS simulator. Full simulator and Android build/navigation verification run in CI and must pass before this stage is accepted. Full Xcode is selected on the macOS CI host. Live staging flows, assistive-technology review and hosted offline/sleep tests remain release gates.
