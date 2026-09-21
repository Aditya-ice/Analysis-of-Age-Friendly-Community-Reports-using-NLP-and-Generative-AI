# Android v2 pilot update

The app now uses invite access, verified completion, keyword fallback and explicit follow-up context. See [shared client setup](../docs/CLIENTS_V2.md). Pilot tokens are encrypted with Android Keystore; all Google credentials remain on the server. Regenerate DTOs and shared test fixtures before building.

# ElderHelp Android prototype

Native single-activity Kotlin/Compose app, minimum Android 8 (API 26). Ask streams answers with cancellation/retry and citation links. Reports are cached in Room; completed answers and citations remain on-device for offline viewing and deletion. No accounts or provider credentials are used. Android backups are disabled for local history.

Use JDK 17 and Android SDK 35. From this directory:

```sh
./gradlew testDebugUnitTest assembleDebug lintDebug
./gradlew installDebug -PelderhelpApiUrl=http://10.0.2.2:8765/
```

Run `python3 ios/MockServer/server.py` from the repository root for the shared mock API. The emulator reaches the host at `10.0.2.2`. Debug builds allow HTTP for local development. Release builds require HTTPS; configure `-PelderhelpApiUrl=https://YOUR-STAGING-API/`. The default is deliberately unconfigured (`https://localhost/`). No deployment or staging credentials are embedded.

Regenerate network models with `python3 android/tools/generate_models.py` from the root. Models are generated from the committed OpenAPI document, including completion/citation schemas. CI checks generated output and runs parser/transport tests, lint, and an APK build.

UI uses system font scaling, text labels for all controls, 48dp minimum citation targets, scrollable screens/sheets, and non-color loading/error status. Validate TalkBack, large fonts, offline history, delete-all confirmation, cancellation, and citation links on an emulator/device before release. Staging integration remains dependent on backend deployment.

The application uses a ViewModel with lifecycle-aware StateFlow collection and a repository over Room and OkHttp. Version pins deliberately use a compatible AGP 8.9 / Gradle 8.11 / Kotlin 2.1 toolchain; upgrades can be isolated after the first verified build.
