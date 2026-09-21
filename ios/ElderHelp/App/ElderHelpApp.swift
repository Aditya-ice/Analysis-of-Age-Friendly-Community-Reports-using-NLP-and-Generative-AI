import SwiftData
import SwiftUI

@main
struct ElderHelpApp: App {
    private let apiClient = APIClient(baseURL: AppConfiguration.apiBaseURL, token: PilotKeychain.read())

    var body: some Scene {
        WindowGroup {
            RootView(apiClient: apiClient)
        }
        .modelContainer(for: [StoredConversation.self, CachedReport.self])
    }
}
