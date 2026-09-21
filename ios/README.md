# Swift v2 pilot update

The app now uses invite access, verified completion, keyword fallback and explicit follow-up context. See [shared client setup](../docs/CLIENTS_V2.md). Pilot tokens are stored in device-only Keychain; all Google credentials remain on the server. The simulator uses the shared v2 mock server and test-only invite.

# ElderHelp iOS

The iOS prototype is a native Swift 6 and SwiftUI application targeting iOS 17. It contains Ask,
Reports, and History tabs, streams cited answers over SSE, caches report metadata, and stores chat
history only on the device with SwiftData.

Generate the checked-in Xcode project after editing `project.yml`:

```sh
cd ios
xcodegen generate
```

Open `ElderHelp.xcodeproj`, select an iOS 17 or newer simulator, and run the `ElderHelp` scheme.
The debug build defaults to `http://127.0.0.1:8000`; change `API_BASE_URL` in `project.yml` for a
staging backend and regenerate the project. No Gemini or Google Cloud credential belongs in the
app.

Full Xcode is required for app and simulator builds. The smaller `Package.swift` exposes the
Foundation-only API contract and SSE parser so they can also be tested with `swift test` when a
matching Swift toolchain and SDK are selected.

CI starts `MockServer/server.py` and runs the unit and UI suites in an iOS simulator. The UI flow
streams deliberately fragmented SSE events, opens a citation sheet, loads Reports, and verifies
that the completed answer appears in on-device History.
