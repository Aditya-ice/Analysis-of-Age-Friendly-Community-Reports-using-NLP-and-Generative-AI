import SwiftUI

struct RootView: View {
    let apiClient: APIClient

    var body: some View {
        TabView {
            NavigationStack {
                AskView(apiClient: apiClient)
            }
            .tabItem { Label("Ask", systemImage: "bubble.left.and.text.bubble.right") }

            NavigationStack {
                ReportsView(apiClient: apiClient)
            }
            .tabItem { Label("Reports", systemImage: "books.vertical") }

            NavigationStack {
                HistoryView()
            }
            .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
        }
    }
}
